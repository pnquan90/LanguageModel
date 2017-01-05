# LanguageModel
A revamp of Recurrent Neural Language Model, to learn new techniques (Weight tying, Bayesian LSTM ...)

Requirement: 

- Torch7, cunn, cudnn (I hardcoded some cudnn modules such as LogSoftMax but apparently they are useless)
- rnn from Element Research https://github.com/Element-Research/rnn : I used their AbstractRecurrent container which is very useful to write recurrent networks.

With the default setting in option.lua: H=650, Dropout = {0.4, 0.25} and dropping LR after 12th Epoch, the script produces test perplexity = 75 on Penntreebank. 

Further practice: Implementing Highway Recurrent Networks, Pointer Networks ...
