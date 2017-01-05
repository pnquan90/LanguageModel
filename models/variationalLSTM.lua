require 'rnn'

local VariationalLSTM, parent = torch.class("nn.VariationalLSTM", "nn.AbstractRecurrent")

-- set this to true to have it use nngraph instead of nn
-- setting this to true can make your next GraphLSTM significantly faster

function VariationalLSTM:__init(inputSize, outputSize, rho, drop_input, drop_hidden, drop_output)
	
	parent.__init(self, rho or 9999)
	self.inputSize = inputSize
	self.outputSize = outputSize
	self.drop_output = drop_output
	print(self.drop_output)
	self.drop_input  = drop_input or 0
	self.drop_hidden = drop_hidden or 0
	
	-- build the model
	self.recurrentModule = self:buildModel()
	
	self.modules[1] = self.recurrentModule
	self.sharedClones[1] = self.recurrentModule
	
	self.zeroTensor = torch.Tensor()
	
	self.cells = {}
	self.gradCells = {}
		
	-- dropout parameters
	self.noise_i = torch.Tensor()
	self.noise_h = torch.Tensor()
	
	if self.drop_output then
		self.noise_o = torch.Tensor()
	end
	
	
end


-- An interface to generate noise masks
-- Supposed to be called before training every sequence
-- Also, the noise must be reset to 1 before testings
function VariationalLSTM:sampleNoise(batch_size)
	self.noise_i:resize(batch_size, self.inputSize):zero()
	self.noise_h:resize(batch_size, self.outputSize):zero()
	
	if self.noise_o then
		self.noise_o:resize(batch_size, self.outputSize):zero()
	end
	
	self.noise_i:bernoulli(1 - self.drop_input)
	self.noise_i:div(1 - self.drop_input)
	
	self.noise_h:bernoulli(1 - self.drop_hidden)
	self.noise_h:div(1 - self.drop_hidden)
	
	if self.noise_o then
		self.noise_o:bernoulli(1 - self.drop_output)
		self.noise_o:div(1 - self.drop_output)
	end
end

function VariationalLSTM:resetNoise(batch_size)
	self.noise_i:resize(batch_size, self.inputSize):zero():add(1)
	self.noise_h:resize(batch_size, self.outputSize):zero():add(1)
	
	if self.noise_o then
		self.noise_o:resize(batch_size, self.outputSize):zero():add(1)
	end
end


function VariationalLSTM:buildModel()
	-- input : {input, prevOutput, prevCell, noise_i, noise_h}
	-- output : {output, cell}
	
	-- Dropout is applied based on Yarin Gal's proposal, but the speed reduction is too much
	-- So I apply directly on the x and h before every gate

	-- Calculate all four gates in one go : input, hidden, forget, output
	self.i2g = nn.Linear(self.inputSize, 4*self.outputSize)
	self.o2g = nn.Linear(self.outputSize, 4*self.outputSize)
	
	
	
	local function local_Dropout(input, noise)
		return nn.CMulTable()({input, noise})
	end

	assert(nngraph, "Missing nngraph package")
	local x, prev_h, prev_c, noise_i, noise_h, noise_o
	local i2h, h2h
	
	
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- x
	table.insert(inputs, nn.Identity()()) -- prev_h
	table.insert(inputs, nn.Identity()()) -- prev_c
	table.insert(inputs, nn.Identity()()) -- noise_i 
	table.insert(inputs, nn.Identity()()) -- noise_h

	if self.drop_output then 
		table.insert(inputs, nn.Identity()()) -- dropout at output layer for the final LSTM layer
		x, prev_h, prev_c, noise_i, noise_h, noise_o = unpack(inputs)
	else
		x, prev_h, prev_c, noise_i, noise_h = unpack(inputs)
	end
	 
	local dropped_x = local_Dropout(x, noise_i)
	local dropped_h = local_Dropout(prev_h, noise_h)
  
	
	-- Calculate 4 gates:
	i2h = self.i2g(dropped_x):annotate{name='i2h'}
    h2h = self.o2g(dropped_h):annotate{name='h2h'}
	
	
	-- Nonlinearity
	local all_input_sums = nn.CAddTable()({i2h, h2h})

	local reshaped = nn.Reshape(4, self.outputSize)(all_input_sums)
	-- input, hidden, forget, output
	local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
	local in_gate = nn.Sigmoid()(n1)
	local in_transform = nn.Tanh()(n2)
	local forget_gate = nn.Sigmoid()(n3)
	local out_gate = nn.Sigmoid()(n4)

	-- perform the LSTM update
	local next_c           = nn.CAddTable()({
	 nn.CMulTable()({forget_gate, prev_c}),
	 nn.CMulTable()({in_gate,     in_transform})
	})
	
	  -- gated cells form the output
	  
	local next_h
	local  temp = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
	
	if self.drop_output then
		next_h = local_Dropout(temp, noise_o)
	else
		next_h = temp
	end

	local outputs = {next_h, next_c}

	--~ nngraph.annotateNodes()

	return nn.gModule(inputs, outputs)

   
end

function VariationalLSTM:updateOutput(input)
	
	-- buffer for the previous states
	local prevOutput, prevCell
	
	-- initialise the first hidden state
	if self.step == 1 then
		prevOutput = self.userPrevOutput or self.zeroTensor
		prevCell = self.userPrevCell or self.zeroTensor
		
		-- prepare for batch size
		if input:dim() == 2 then
			self.zeroTensor:resize(input:size(1), self.outputSize):zero()
		else
			self.zeroTensor:resize(self.outputSize):zero()
		end
	else
	
		prevOutput = self.outputs[self.step-1]
		prevCell = self.cells[self.step-1]
	end
	
	
	-- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
	local output, cell
	local inputTable = {input, prevOutput, prevCell, self.noise_i, self.noise_h}
	
	if self.drop_output then
		table.insert(inputTable, self.noise_o)
	end
	
	if self.train ~= false then
		self:recycle()
		
		-- get the clone at the time step
		local recurrentModule = self:getStepModule(self.step)
		
		output, cell = unpack(recurrentModule:updateOutput(inputTable))
		
	else
	
		-- during testing we don't need to store immediate tensors
		output, cell = unpack(self.recurrentModule:updateOutput(inputTable))
	
	end
	
	self.outputs[self.step] = output
	self.cells[self.step]   = cell
	
	self.output = output
	self.cell = cell
	
	self.step = self.step + 1
	
	self.gradPrevOutput = nil
	self.updateGradInputStep = nil
	self.accGradParametersStep = nil
	
	-- return the hidden layer
	return self.output
	
end


function VariationalLSTM:updateGradInput(input, gradOutput)

	self.updateGradInputStep = self.updateGradInputStep or self.step

	assert(self.step > 1, "expecting at least one updateOutput")
	local step = self.updateGradInputStep - 1
	assert(step >= 1)
	
	-- set the output/gradOutput states of current Module
	
	local recurrentModule = self:getStepModule(step)
	
	-- backward propagate through this step
	if self.gradPrevOutput then
	
		-- the hidden layer receives two signals of gradients: from the connection directly after the hidden layer, and the connection to the next RNN step
		-- so we need to sum them up
		self._gradOutputs[steps] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
		nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
		gradOutput = self._gradOutputs[step]
	end
	
	local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
	local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
	
	local inputTable = {input, output, cell, self.noise_i, self.noise_h}
	if self.drop_output then
		table.insert(inputTable, self.noise_o)
	end
	
    local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
    
    local gradInputTable = recurrentModule:updateGradInput(inputTable, {gradOutput, gradCell})
    
    local gradInput
    --~ gradInput, self.prevGradOutput, gradCell,  = unpack(gradInputTable)
    gradInput = gradInputTable[1]
    self.prevGradOutput = gradInputTable[2]
    gradCell = gradInputTable[3]
    self.gradCells[step-1] = gradCell
    
    if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
    if self.userPrevCell   then self.userGradPrevCell   = gradCell            end
    
    self.gradInput = gradInput
    
    self.updateGradInputStep = self.updateGradInputStep - 1
    self.gradInputs[self.updateGradInputStep] = self.gradInput
    
    return self.gradInput
    
	
end


function VariationalLSTM:accGradParameters(input, gradOutput, scale)
	
	assert(self.updateGradInputStep < self.step, "Missing updateGradInput")
	self.accGradParametersStep = self.accGradParametersStep or self.step
	
	local step = self.accGradParametersStep - 1 
	assert(step >= 1)
	
	-- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)
   
   -- backward propagate through this step
    local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
    local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
    local inputTable = {input, output, cell, self.noise_i, self.noise_h}
	if self.drop_output then
		table.insert(inputTable, self.noise_o)
	end
    local gradOutput = (step == self.step-1) and gradOutput or self._gradOutputs[step]
   
    local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
   
    local gradOutputTable = {gradOutput, gradCell}
   
    recurrentModule:accGradParameters(inputTable, gradOutputTable, scale)
	
	self.accGradParametersStep = self.accGradParametersStep - 1

end


function VariationalLSTM:clearState()
   self.zeroTensor:set()
   self.step = 1
   self.noise_i:set()
   self.noise_h:set()
   if self.noise_o then self.noise_o:set() end
   return parent.clearState(self)
end

function VariationalLSTM:type(type, ...)
   if type then
      self:forget()
      self:clearState()
      self.zeroTensor = self.zeroTensor:type(type)
   end
   return parent.type(self, type, ...)
end
