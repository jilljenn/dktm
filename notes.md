It's a bit a pain in the fromage to learn PyTorch.

# Handling sequences of different lengths

I started with the only starting page, but it was clearly not enough.

https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-control-flow-weight-sharing

In the particular, I wanted to find a way to handle long sequences, and I had problems of `retain_graph` parameter when calling `loss.backward()`.

https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795

But actually, it was because I was running the code on the full dataset, and was calling `loss.backward()` multiple times.

Also these kinds of solutions do not help when you don't understand how PyTorch works.

https://discuss.pytorch.org/t/lstm-how-to-handle-long-input-sequences-with-fixed-length-bptt/3460

So you have to go through the simplest questions you can ask:

- https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350/1
- https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944/1

But finally, maybe the most useful example was:

https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20

Where they do show the difference in terms of memory according to where you `loss.backward()` and where you `optimizer.step()`.

Also this code was useful, thanks to the comments:

https://github.com/pytorch/examples/blob/master/word_language_model/main.py

Even if I'm not sure what they mean by cutting one huge sequence into several batches (the alphabet example).

Maybe for BPTT, the following resources will also help:

- https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500
- https://discuss.pytorch.org/t/how-to-train-a-many-to-many-lstm-with-bptt/13005
- https://github.com/kanekomasahiro/SLAM18_model/blob/master/trainRNNLM.py#L228
