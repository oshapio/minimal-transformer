
Dependence-free original Transformer [1] re-implementation. The main goal here was to understand the inner-workings of the architecture rather than to build a production-ready efficient system. As such, I focus on simplicity and ease of understanding, which hopefully serve useful for others that want to see how the information flows in the architecture without the need to spend much effort processing data. 

Two modes of usage are available (controlled by `trainer.py`):
- Classification mode (`task = classification`). Here only the encoder is used, whose representations are averaged out and a classification is made. I consider a task to classify whether the first element in the sequence is the identical to the last one. 
- Seq2seq mode (`task = seq2seq`). This is the task considered in the original paper, and is way more involved than the clasification one. I consider a sequence reversal task.   

Some TODOs:
- [ ] Properly test the model (especially the sequence-decoder module)
- [ ] Add dropout support
- [ ] Add support for output sequences that differ in feature dimensionality and length from the input sequences.
- [ ] Add weight multiplication of the Linear layer.

---

[1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).