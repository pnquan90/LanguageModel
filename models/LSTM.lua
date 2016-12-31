require 'rnn'

local baseLSTM, parent = torch.class("nn.baseLSTM", "nn.AbstractRecurrent")

-- set this to true to have it use nngraph instead of nn
-- setting this to true can make your next GraphLSTM significantly faster

function baseLSTM:__init(inputSize, outputSize, rho)
	
	parent.__init(self, rho or 9999)
	self.inputSize = inputSize
	self.outputSize = outputSize
	self.bn  = false
	self.eps = 0.1
	self.momentum = 0.1
	self.affine = true
	
	-- build the model
	self.recurrentModule = self:buildModel()
	
	self.modules[1] = self.recurrentModule
	self.sharedClones[1] = self.recurrentModule
	
	self.zeroTensor = torch.Tensor()
	
	self.cells = {}
	self.gradCells = {}
	
	
end
--~ GraphLSTM.usenngraph = true
--~ GraphLSTM.bn = false


function baseLSTM:buildModel()
   -- input : {input, prevOutput, prevCell}
   -- output : {output, cell}
   
   -- Calculate all four gates in one go : input, hidden, forget, output
   self.i2g = nn.Linear(self.inputSize, 4*self.outputSize)
   self.o2g = nn.Linear(self.outputSize, 4*self.outputSize)
   
   assert(nngraph, "Missing nngraph package")
   
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x
   table.insert(inputs, nn.Identity()()) -- prev_h[L]
   table.insert(inputs, nn.Identity()()) -- prev_c[L]
   
   local x, prev_h, prev_c = unpack(inputs)

   local bn_wx, bn_wh, bn_c  
   local i2h, h2h 
   if self.bn then  
      -- apply recurrent batch normalization 
      -- http://arxiv.org/pdf/1502.03167v3.pdf
      -- normalize recurrent terms W_h*h_{t-1} and W_x*x_t separately 
      -- Olalekan Ogunmolu <patlekano@gmail.com>
   
      bn_wx = nn.BatchNormalization(4*self.outputSize, self.eps, self.momentum, self.affine)
      bn_wh = nn.BatchNormalization(4*self.outputSize, self.eps, self.momentum, self.affine)
      bn_c  = nn.BatchNormalization(self.outputSize, self.eps, self.momentum, self.affine)
      
      -- evaluate the input sums at once for efficiency
      i2h = bn_wx(self.i2g(x):annotate{name='i2h'}):annotate {name='bn_wx'}
      h2h = bn_wh(self.o2g(prev_h):annotate{name='h2h'}):annotate {name = 'bn_wh'}
      
      -- add bias after BN as per paper
      self.o2g:noBias()
      h2h = nn.Add(4*self.outputSize)(h2h)
   else
      -- evaluate the input sums at once for efficiency
      i2h = self.i2g(x):annotate{name='i2h'}
      h2h = self.o2g(prev_h):annotate{name='h2h'}
   end
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
   local next_h
   if self.bn then
      -- gated cells form the output
      next_h = nn.CMulTable()({out_gate, nn.Tanh()(bn_c(next_c):annotate {name = 'bn_c'}) })
   else
      -- gated cells form the output
      next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
   end

   local outputs = {next_h, next_c}

   nngraph.annotateNodes()
   
   return nn.gModule(inputs, outputs)
   
   
end

function baseLSTM:updateOutput(input)
	
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
	
	if self.train ~= false then
		self:recycle()
		
		-- get the clone at the time step
		local recurrentModule = self:getStepModule(self.step)
		
		output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
		
	else
	
		-- during testing we don't need to store immediate tensors
		output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
	
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


function baseLSTM:updateGradInput(input, gradOutput)

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
	
	local inputTable = {input, output, cell}
    local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
    
    local gradInputTable = recurrentModule:updateGradInput(inputTable, {gradOutput, gradCell})
    
    local gradInput 
    gradInput, self.prevGradOutput, gradCell = unpack(gradInputTable)
    self.gradCells[step-1] = gradCell
    
    if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
    if self.userPrevCell   then self.userGradPrevCell   = gradCell            end
    
    self.gradInput = gradInput
    
    self.updateGradInputStep = self.updateGradInputStep - 1
    self.gradInputs[self.updateGradInputStep] = self.gradInput
    
    return self.gradInput
    
	
end


function baseLSTM:accGradParameters(input, gradOutput, scale)
	
	assert(self.updateGradInputStep < self.step, "Missing updateGradInput")
	self.accGradParametersStep = self.accGradParametersStep or self.step
	
	local step = self.accGradParametersStep - 1 
	assert(step >= 1)
	
	-- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)
   
   -- backward propagate through this step
    local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
    local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
    local inputTable = {input, output, cell}
    local gradOutput = (step == self.step-1) and gradOutput or self._gradOutputs[step]
   
    local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
   
    local gradOutputTable = {gradOutput, gradCell}
   
    recurrentModule:accGradParameters(inputTable, gradOutputTable, scale)
	
	self.accGradParametersStep = self.accGradParametersStep - 1

end


--~ function GraphLSTM:nngraphModel()
   --~ assert(nngraph, "Missing nngraph package")
   
   --~ local inputs = {}
   --~ table.insert(inputs, nn.Identity()()) -- x
   --~ table.insert(inputs, nn.Identity()()) -- prev_h[L]
   --~ table.insert(inputs, nn.Identity()()) -- prev_c[L]
   
   --~ local x, prev_h, prev_c = unpack(inputs)

   --~ local bn_wx, bn_wh, bn_c  
   --~ local i2h, h2h 
   --~ if self.bn then  
      --~ -- apply recurrent batch normalization 
      --~ -- http://arxiv.org/pdf/1502.03167v3.pdf
      --~ -- normalize recurrent terms W_h*h_{t-1} and W_x*x_t separately 
      --~ -- Olalekan Ogunmolu <patlekano@gmail.com>
   
      --~ bn_wx = nn.BatchNormalization(4*self.outputSize, self.eps, self.momentum, self.affine)
      --~ bn_wh = nn.BatchNormalization(4*self.outputSize, self.eps, self.momentum, self.affine)
      --~ bn_c  = nn.BatchNormalization(self.outputSize, self.eps, self.momentum, self.affine)
      
      --~ -- evaluate the input sums at once for efficiency
      --~ i2h = bn_wx(self.i2g(x):annotate{name='i2h'}):annotate {name='bn_wx'}
      --~ h2h = bn_wh(self.o2g(prev_h):annotate{name='h2h'}):annotate {name = 'bn_wh'}
      
      --~ -- add bias after BN as per paper
      --~ self.o2g:noBias()
      --~ h2h = nn.Add(4*self.outputSize)(h2h)
   --~ else
      --~ -- evaluate the input sums at once for efficiency
      --~ i2h = self.i2g(x):annotate{name='i2h'}
      --~ h2h = self.o2g(prev_h):annotate{name='h2h'}
   --~ end
   --~ local all_input_sums = nn.CAddTable()({i2h, h2h})

   --~ local reshaped = nn.Reshape(4, self.outputSize)(all_input_sums)
   --~ -- input, hidden, forget, output
   --~ local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
   --~ local in_gate = nn.Sigmoid()(n1)
   --~ local in_transform = nn.Tanh()(n2)
   --~ local forget_gate = nn.Sigmoid()(n3)
   --~ local out_gate = nn.Sigmoid()(n4)
   
   --~ -- perform the LSTM update
   --~ local next_c           = nn.CAddTable()({
     --~ nn.CMulTable()({forget_gate, prev_c}),
     --~ nn.CMulTable()({in_gate,     in_transform})
   --~ })
   --~ local next_h
   --~ if self.bn then
      --~ -- gated cells form the output
      --~ next_h = nn.CMulTable()({out_gate, nn.Tanh()(bn_c(next_c):annotate {name = 'bn_c'}) })
   --~ else
      --~ -- gated cells form the output
      --~ next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
   --~ end

   --~ local outputs = {next_h, next_c}

   --~ nngraph.annotateNodes()
   
   --~ return nn.gModule(inputs, outputs)
--~ end

--~ function GraphLSTM:buildGate()
   --~ error"Not Implemented"
--~ end

--~ function GraphLSTM:buildInputGate()
   --~ error"Not Implemented"
--~ end

--~ function GraphLSTM:buildForgetGate()
   --~ error"Not Implemented"
--~ end

--~ function GraphLSTM:buildHidden()
   --~ error"Not Implemented"
--~ end

--~ function GraphLSTM:buildCell()
   --~ error"Not Implemented"
--~ end   
   
--~ function GraphLSTM:buildOutputGate()
   --~ error"Not Implemented"
--~ end

function baseLSTM:clearState()
   self.zeroTensor:set()
   self.step = 1
   return parent.clearState(self)
end

function baseLSTM:type(type, ...)
   if type then
      self:forget()
      self:clearState()
      self.zeroTensor = self.zeroTensor:type(type)
   end
   return parent.type(self, type, ...)
end
