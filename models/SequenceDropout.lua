--~ Dropout at sequence level 
--~ To be used after embedding layer
--~ Dropout detail is described in Gal et al (2015) 
--~ This feature has already been used in Nematus (shared dropout layer at embedding level)

local SequenceDropout, Parent = torch.class('nn.SequenceDropout', 'nn.Module')

function SequenceDropout:__init(p, tying, inplace)
	
	Parent.__init(self)
	self.p = p
	self.train = true
	self.inplace = inplace
	self.tying = tying or false
	
	self.noise = torch.Tensor()
	
end	


function SequenceDropout:updateOutput(input)
	
	-- Require input of size: N-time_step * batch_size * input_size
	-- Inplace dropout saves memory
	if self.inplace then
		self.output:set(input)
	else
		self.output:resizeAs(input):copy(input)
	end
	
	if self.p > 0 then
		
		if self.train then
			self.noise:resizeAs(input):zero()
			
			local T = self.noise:size(1) -- n time steps
			local B = self.noise:size(2) -- batch_size
			local input_size = self.noise:size(3)
						
			
			for b = 1, B do
				for t = 1, T do
					-- generate noise for each time step
					self.noise[t][b]:bernoulli(1 - self.p)
					self.noise[t][b]:div(1 - self.p)
					
					if self.tying == true then
						-- tying dropout over words 
						local x = input[t][b]
						for j = t+1, T do
							if input[j][b] == x then
								self.noise[j][b] = self.noise[t][b]
							end
						end
					end
				
				end
			end
			
			
			self.output:cmul(self.noise)
		end
		
		
	end
	
	
	
	return self.output

end

function SequenceDropout:updateGradInput(input, gradOutput)
	
	if self.inplace then
		self.gradInput:set(gradOutput)
	else
		self.gradInput:resizeAs(gradOutput):copy(gradOutput)
	end
	
	if self.train then
		if self.p > 0 then
			self.gradInput:cmul(self.noise)
		end
	end
	
	return self.gradInput
end 


function SequenceDropout:setp(p)
   self.p = p
end

function SequenceDropout:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end


function SequenceDropout:clearState()
   if self.noise then
      self.noise:set()
   end
   
   self.noiseTable = {}
   
   return Parent.clearState(self)
end
