require 'models.LSTM'
require 'models.graphLSTM'
require 'models.variationalLSTM'
require 'models.variationalSequencer'
require 'models.SequenceDropout'

local LM = torch.class('nn.RecurrentLanguageModel')


function LM:__init(params, vocab_size)

	self.n_hiddens = params.n_hidden
	self.n_layers = params.n_layers
	self.dropout = params.dropout
	self.initial_val = params.initial_val
	self.vocab_size = vocab_size
	self.gradient_clip = params.gradient_clip
	
	--~ Model definition
	self.net = nn.Sequential()
	self.rnns = {}
	
	local V, H = self.vocab_size, self.n_hiddens
	--~ self.net:add(nn.LookupTable(V, H))
	local lut = nn.LookupTable(V, H)
	self.net:add(lut)
	
	-- Dropout on top of embedding
	self.net:add(nn.SequenceDropout(params.drop_emb))
	
	for i = 1, self.n_layers do

		local LSTM, sequenceLSTM
		if i < self.n_layers then
			LSTM = nn.VariationalLSTM(H, H, 9999, params.drop_input, params.drop_hidden)
		else
			LSTM = nn.VariationalLSTM(H, H, 9999, params.drop_input, params.drop_hidden, params.drop_output)
		end
		
		sequenceLSTM = nn.VariationalSequencer(LSTM)
		
		self.net:add(sequenceLSTM)
		
		--~ stepModule:add(nn.Dropout(self.dropout))
		--~ local rnn = nn.baseLSTM(H, H)
		--~ local rnn = nn.graphLSTM(H, H)
		--~ stepModule:add(rnn)
		
		--~ table.insert(self.rnns, rnn)
	end
	
	--~ self.net:add(self.sequencer)
	--~ self.net:add(nn.Dropout(self.dropout))	
	
	local linear = nn.Linear(H, V)
	local softmax = cudnn.LogSoftMax()
	
	self.net:add(nn.Sequencer(linear))
	self.net:add(nn.Sequencer(softmax))
	
	--~ Ship model to GPU
	--~ Does not support CPU due to using cudnn
	self.net:cuda()
	
	linear:share(lut, 'weight', 'gradWeight')
	self.net:remember('both')
	
	
	
	self.params, self.gradParams = self:getParameters()
	--~ self.best_params = torch.CudaTensor(self.params:nElement()):copy(self.params)
	self.params:uniform(-self.initial_val, self.initial_val)
	
	
	
	local total_params = self.params:nElement()
	print("Total parameters of model: " .. total_params)
	
	local crit = nn.ClassNLLCriterion(nil, true):cuda()
	
	self.criterion = nn.SequencerCriterion(crit)
	
	print(self.net)
	
	
end

-- Evaluation function: Compute NLL of the validation set
function LM:eval(input, target)
	
	local N, T = input:size(1), input:size(2)
	local n_samples = N 
	
	
	local net_output = self.net:forward(input)
		
	local loss = self.criterion:forward(net_output, target)
	
	return loss, n_samples
end

function LM:trainBatch(input, target, learning_rate)
	
	self.net:zeroGradParameters()
	
	local N, T = input:size(1), input:size(2)
	
	local batch_size = T
	local n_samples = N 
	
	--~ Forward Pass	
	local net_output = self.net:forward(input)
	
	
	local loss = self.criterion:forward(net_output, target)
	
		
	local gradloss = self.criterion:backward(net_output, target)
	
	
	self.net:backward(input, gradloss)
	
	norm = self.gradParams:norm()

	-- gradient clipping
	if norm > self.gradient_clip then
		self.gradParams:mul(self.gradient_clip / norm)
	end	
	
	self.net:updateParameters(learning_rate)
	collectgarbage()
	
	return loss , n_samples
end


function LM:getParameters()
	return self.net:getParameters()
end

function LM:training()
	self.net:training()
end

function LM:evaluate()
	self.net:evaluate()
end

function LM:clearState()
	self.net:clearState()
	--~ self.sequencer:clearStates()
end

