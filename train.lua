
-- This file trains and tests the RNN from a batch loader.
require('nn')
require('options')
require 'utils.misc'
require 'cudnn'
require 'models.RecurrentLanguageModel'
require 'nngraph'

model_utils = require('utils.model_utils')
require('utils.batchloader')
require('utils.textsource')

-- Parse arguments
local cmd = RNNOption()
g_params = cmd:parse(arg)
torch.manualSeed(1990)

require 'cutorch'
require 'cunn'
require 'rnn'
cutorch.setDevice(g_params.cuda_device)
cutorch.manualSeed(2056)
cudnn.fastest = true
cudnn.benchmark = true

local dropout_params = {drop_input = 0.4, drop_hidden = 0.2, drop_output = 0.4, drop_emb = 0.2}
g_params.model.drop_input = dropout_params.drop_input
g_params.model.drop_hidden = dropout_params.drop_hidden
g_params.model.drop_output = dropout_params.drop_output
g_params.model.drop_emb    = dropout_params.drop_emb

cmd:print_params(g_params)




-- build the torch dataset
local g_dataset = TextSource(g_params.dataset)
local vocab_size = g_dataset:get_vocab_size()
local g_dictionary = g_dataset.dict
-- A data sampler for training and testing
batch_loader = BatchLoader(g_params.dataset, g_dataset)
local word2idx = g_dictionary.symbol_to_index
local idx2word = g_dictionary.index_to_symbol

local x, y = batch_loader:next_batch(2)
local N, T = x:size(1), x:size(2)

local unigrams = g_dataset.dict.index_to_freq:clone()
unigrams:div(unigrams:sum())


local model = nn.RecurrentLanguageModel(g_params.model, vocab_size)








local function eval(split_id)

	model:evaluate()
	split_id = split_id or 2
	
	local x, y = batch_loader:next_batch(split_id)
	local N, T = x:size(1), x:size(2)
	--~ print(N, T)
	local total_loss = 0
	local total_samples = 0
	
	--note that batch size in eval is different to batch size in training (small)
	local index = 1
	
	while index <= N do
		
		local stop = math.min(N, index + 16)
		xlua.progress(stop, N)
		local x_seq = x:sub(index, stop, 1, -1)
		local y_seq = y:sub(index, stop, 1, -1)
		
		local loss, n_samples = model:eval(x_seq, y_seq)
		total_loss = total_loss + loss
		total_samples = total_samples + n_samples
		index = index + 17
	end
	

	
	local avg_loss = total_loss / total_samples
	local ppl = torch.exp(avg_loss)
	
	model:clearState()

	return avg_loss
end


local function train_epoch(learning_rate, batch_size)
	model:training()
		
	batch_loader:reset_batch_pointer(1)
	--~ model:createHiddenInput(batch_size)
	
	local speed
	local n_batches = batch_loader.split_sizes[1]
	local total_loss = 0
	local total_samples = 0

	local timer = torch.tic()
	
	for i = 1, n_batches do
		
		xlua.progress(i, n_batches)

		-- forward pass 
		local input, target = batch_loader:next_batch(split)
		
		local loss, n_samples = model:trainBatch(input, target, learning_rate)
		total_loss = total_loss + loss
		total_samples = total_samples + n_samples
		
	end

	local elapse = torch.toc(timer)
	

	total_loss = total_loss / total_samples

	local perplexity = torch.exp(total_loss)
	
	collectgarbage()
	
	local speed = math.floor(total_samples * batch_size / elapse)
	
	model:clearState()

	return total_loss, speed

end

local function run(n_epochs, anneal)

	anneal = anneal or 'fast'
	
	print('Annealing: ' .. anneal)

	--~ print(torch.exp(eval(3)))
	local val_loss = {}
	local l = eval(2)
	print(torch.exp(l))
	val_loss[0] = l

	local patience = 0

	local learning_rate = g_params.trainer.initial_learning_rate
	local batch_size = g_params.trainer.batch_size
	
	--~ 
	
	for epoch = 1, g_params.trainer.max_max_epochs do
		
		-- Early stopping after a long time no improvement on Valid set
		if patience > g_params.trainer.max_patience then
			break
		end 
		
		local train_loss, wps = train_epoch(learning_rate, batch_size)

		
		val_loss[epoch] = eval(2)
	
		
		if anneal == 'fast' then
		-- this schedule is from Zaremba to converge faster
			if epoch >= g_params.trainer.max_epochs then
				
				learning_rate = learning_rate * g_params.trainer.learning_rate_shrink
			
			end
		
		else
			-- Control patience when no improvement
			if val_loss[epoch] >= val_loss[epoch - 1] * g_params.trainer.shrink_factor then
				learning_rate = learning_rate * g_params.trainer.learning_rate_shrink
			end
		
		end
		
		-- long time no improvement -> increase patience
		if val_loss[epoch] >= val_loss[epoch - 1] * g_params.trainer.shrink_factor then
			patience = patience + 1
		else
			patience = 0
		end
		
		
		--~ Display training information
		local stat = {train_perplexity = torch.exp(train_loss) , epoch = epoch,
                valid_perplexity = torch.exp(val_loss[epoch]), LR = learning_rate, speed = wps, patience = patience}

        print(stat)
        
        -- save the trained model
		--~ local save_dir = g_params.trainer.save_dir
		--~ if save_dir ~= nil then
		  --~ if paths.dirp(save_dir) == false then
			  --~ os.execute('mkdir -p ' .. save_dir)	
		  --~ end
		  --~ local save_state = {}
		  --~ save_state.model = model
		  --~ save_state.criterion = criterion
		  --~ save_state.learning_rate = learning_rate
		  --~ torch.save(paths.concat(save_dir, 'model_' .. epoch), save_state)
		--~ end

        -- early stop when learning rate too small
        if learning_rate <= 1e-3 then break end
        
		

	end
	
	print(torch.exp(eval(3)))
			

	
	
end


g_params.trainer.shrink_factor = 0.9999
run(g_params.trainer.n_epochs, 'fast')



