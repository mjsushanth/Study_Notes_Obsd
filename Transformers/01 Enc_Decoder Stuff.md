
  
Architectures and training methods vary widely between LLMs (and SLMs). But what most modern language models have in common is that they are based on the transformer architecture. And when it comes to pre-training modern LLMs, there are two key architectures that stand out: encoder-based models and decoder-based models. They’re each good at different types of tasks and that’s because they are trained differently.  
  
⭐️ Encoder-based models focus on language understanding and generally tend to be smaller. These models are trained on Masked Language Modeling (MLM) tasks. Here’s how it works:  
  
1. A percentage of the input tokens in the sequence are randomly blanked out.  
The encoder processes the corrupted sequence, generating contextual embeddings for every token, including the masked ones.  
2. The model must predict the original, masked tokens based only on the surrounding context provided by the unmasked tokens and the generated embeddings.  
3. Cross-Entropy Loss is calculated by measuring the dissimilarity between the model’s predicted word probability and the true original word.  
➪ The model is trained to minimize the loss function, learning a bidirectional understanding of the text (because it can use the tokens in front of AND after the masked token to predict what the masked token should be).  
  
⭐️ Decoder-based models are focused on language generation and are definitely on the larger side (think billions to trillions of parameters). These models are trained on Causal Language Modeling (CLM) tasks. Here’s how it works:  
  
4. The model is fed a sequence of tokens from the training text, and the attention mechanism is restricted so that each token can only look backward at preceding tokens.  
5. The model is trained to perform Next Token Prediction: predicting the most probable next token in the sequence, using only the preceding tokens to guide it.  
6. Cross-Entropy Loss is calculated for each prediction against the true subsequent token.  
➪ The process trains the decoder to develop an autoregressive, unidirectional understanding essential for sequential text generation.  
  
Learning how a model is trained is imperative to understanding which tasks it’s best suited to complete. These two blog posts on encoder vs. decoder models are a great place to start.  
  
Encoder vs. Decoder vs. Encoder-Decoder Models [Aman Chadha]  
🔗: [https://lnkd.in/g5bm3hhc](https://lnkd.in/g5bm3hhc)  
  
Understanding Encoder And Decoder LLMs [Sebastian Raschka]  
🔗: [https://lnkd.in/gjkvXi68](https://lnkd.in/gjkvXi68)