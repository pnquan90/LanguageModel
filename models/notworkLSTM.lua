

--~ [[ LSTM - Long-short term memory architecture ]]

--~ [[ The purpose of this implementation is to understand more about LSTM mechanism ]]

--~ [[ As well as expanding LSTM or any recurrent module in the future ]]


-- Expects 3D inputs:    nsteps x mbsize x isize
-- Outputs 3D outputs:   nsteps x mbsize x hsize
-- The intermediate hidden and cell layers are stored during computation
require 'utils.model_utils'

local qLSTM, parent = torch.class('nn.qLSTM', 'nn.Module')


function qLSTM:__init(input_size, hidden_size, p)
	
	p = p or 0 -- dropout value
	self.input_size = input_size
	self.hidden_size = hidden_size
	--~ print(self.input_size, self.hidden_zie)
	-- Baseline dropout
	self.dropout = p
	
	--~ print(nn.Identity()())
	
	-- Containers for immediate tensors
	self.inputHidden = torch.CudaTensor() -- initial hidden layer
	self.inputCell = torch.CudaTensor()   -- initial cell layer
	self.outputs = torch.CudaTensor()
	self.gradOutputs = torch.CudaTensor()
	self.output = torch.CudaTensor()
	self.gradOutput = torch.CudaTensor()
	self.gradInput = torch.CudaTensor()
	self.gradCell = torch.CudaTensor()
	
	self.tableGradInput = {}
	self.tableGradCell = {}
	self.tableGradHidden = {}
	
	--~ self.protos = {}
	--~ self.protos.rnns = self:buildStep()
	self.core = self:buildStep()
	
	current_num_steps = 0  -- How many steps the network has rolled out. If the new input has less steps then we don't have to re-rollout
	current_index     = 0  -- The index of current step in the rolled netwo
	
	self.params, self.gradParams = self:getParameters()
	
end




-- This function builds the computational flow for one LSTM step
function qLSTM:buildStep()
	
	local inputs = {}
	local outputs = {}
    table.insert(inputs, nn.Identity()()) -- x
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
   
    local x, prev_h, prev_c = unpack(inputs)
	
	if self.dropout > 0 then
		x = nn.Dropout(self.dropout)(x)
	end
		
	-- Build all gates in one go 
	local i2h = nn.Linear(self.input_size, 4 * self.hidden_size)(x):annotate{name='i2h'}
    local h2h = nn.Linear(self.hidden_size, 4 * self.hidden_size)(prev_h):annotate{name='h2h'}
		
	local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, self.hidden_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, cudnn.Tanh()(next_c)})
    
    table.insert(outputs, next_h)
    table.insert(outputs, next_c)
    
    return nn.gModule(inputs, outputs)
   
end


-- Expect input with size: nstep * bsize * inputSize
function qLSTM:updateOutput(input)
	
	
	local nsteps = input:size(1)
	local B = input:size(2)
	self.output:resize(nsteps, B, self.hidden_size)
	
	--~ print(self.output:size())
	-- The current sequence is longer than the current unrolled network 
	--~ if nstep > self.current_num_steps then
		--~ self:unroll(nsteps)
	--~ end
	
	-- Initial hidden state: 0 or remember from the last one
	if self.inputHidden:nElement() == 0 then
		self.inputHidden:resize(B, self.hidden_size):zero()
	end
	
	-- Similar for initial Cell state
	if self.inputCell:nElement() == 0 then
		self.inputCell:resize(B, self.hidden_size):zero()
	end
	
	-- This is only for training, for testing it could be memory insufficient
	-- Solution similar to cudnn: using short sequences during testing 
	self.hiddenStates = {[0] = {self.inputHidden, self.inputCell}}
	
	--~ print(self.hiddenStates)
	
	-- Looping
	for t = 1,nsteps do
		-- the current state is the output of the network given the input and the previous state
		--~ print(input[t]:size())
		--~ print(unpack(self.hiddenStates[t-1]))
		
		local rnnOutput = self.core:forward({input[t], unpack(self.hiddenStates[t-1])})
		self.hiddenStates[t] = {}
		self.hiddenStates[t][1] = rnnOutput[1]:clone()
		self.hiddenStates[t][2] = rnnOutput[2]:clone()
		self.output[t]:copy(self.hiddenStates[t][1])
			
	end
	
	self.current_num_steps = nsteps
	
	
	return self.output
	 
	
end



-- Compute dL/dx given dL/dY
-- This function should be called after updateOutput to ensure hiddenStates consistency
function qLSTM:updateGradInput(input, gradOutput)

	local nsteps = input:size(1)
	--~ assert(nsteps == self.current_num_steps)
	local B = input:size(2)
	
	
	
	--~ self.core:zeroGradParameters()
	
	--~ self.gradCell:resize(gradOutput:size()):fill(0)
	self.gradInput:resize(nsteps, B, self.input_size):fill(0)
	
	
	-- Looping
	--~ self.tableGradCell
	--~ self.tableGradHidden[nsteps] = gradOutput[nsteps]
	for t = 1, nsteps do
		self.tableGradCell[t] = torch.CudaTensor(B, self.hidden_size):fill(0)
		self.tableGradHidden[t] = torch.CudaTensor(B, self.hidden_size):fill(0):copy(gradOutput[t])
	end
	
	for t = nsteps, 1, -1 do
		
		--~ print ('before', self.hiddenStates[t-1][1]:norm()) 
		local gradInputStep = self.core:updateGradInput({input[t], unpack(self.hiddenStates[t-1])}, {self.tableGradHidden[t], self.tableGradCell[t]})
		self.gradInput[t]:copy(gradInputStep[1]) -- for the node input
		
		if t > 1 then
			--~ self.gradCell[t-1]:add(gradInputStep[3])
			self.tableGradCell[t-1]:add(gradInputStep[3])
			self.tableGradHidden[t-1]:add(gradInputStep[2]) 
		end
	end
	
	--~ print(self.gradInput:norm())
	return self.gradInput
	
end


-- Compute dL/dW from dL/dY 
function qLSTM:accGradParameters(input, gradOutput, scale)
	
	local nsteps = input:size(1)
	assert(nsteps == self.current_num_steps)
	--~ self.gradCell:resize(gradOutput:size()):fill(0)
	
	for t = nsteps, 1, -1 do
		
		--~ print(t, self.hiddenStates[t-1][2]:norm())
		
		
		--~ self.gradInput:zero()
		self.core:accGradParameters({input[t], unpack(self.hiddenStates[t-1])}, {self.tableGradHidden[t], self.tableGradCell[t]}, scale)
		--~ print(t, self.gradParams:norm(), self.hiddenStates[t-1][1]:norm())
	end
	
end


function qLSTM:carryHiddenUnits()
	
	--~ print('hihihihi')
	self.inputHidden:copy(self.hiddenStates[#self.hiddenStates][1]) -- 1 for next_h
	self.inputCell:copy(self.hiddenStates[#self.hiddenStates][2])   -- 2 for next_c
end


-- Return weight and gradWeight
function qLSTM:getParameters()
	
	return model_utils.combine_all_parameters(self.core)
	--~ return self.core:getParameters()
end


function qLSTM:training()
	self.core:training()
end

function qLSTM:evaluate()
	self.core:evaluate()
end


function qLSTM:unroll()

end


