# HW to Chapter 19 “Recurrent Neural Networks”

## Non-programming Assignment

### Q1. What are recurrent neural networks (RNN) and why are they needed?     

### Answer:        

Recurrent Neural Networks (RNNs) are a class of artificial neural networks designed to recognize patterns in sequences of data, such as text, genomes, handwriting, or numerical time series data from market stocks and sensors. Unlike traditional neural networks, which process inputs independently, RNNs have loops within them, allowing information to persist. This looping mechanism enables RNNs to take not only the current input but also what they have perceived previously in time into account, effectively giving them a memory.

RNNs are needed for several key reasons:

1. Handling Sequential Data:       
They are inherently designed for sequential data. For tasks like language translation, voice recognition, and text generation, the order and context of the data are crucial for making accurate predictions or generating coherent outputs. RNNs can process sequences of variable length, which is common in real-world data.

2. Temporal Dependencies:        
RNNs can capture temporal dynamic behavior. This makes them suitable for applications such as time series analysis where understanding historical data, trends, and patterns over time is critical for forecasting future events.

3. Contextual Understanding:          
In tasks like language modeling, the meaning of a word can depend heavily on the words that come before or after it. RNNs' ability to maintain information about previous inputs allows them to understand context and subtleties in language, making them powerful tools for NLP tasks.

4. Compact Modeling of Sequences:            
Instead of needing a model size that grows with the length of the input sequence, RNNs can model very long sequences while maintaining a relatively compact representation. This efficiency makes them scalable and practical for large datasets and complex problems.

Overall, RNNs are an essential tool for dealing with the temporal or sequential nature of data across various domains, providing a foundation for building sophisticated models that can learn and make predictions about sequential data.

### Q2. What do time steps play in recurfrent neural networks?

### Answer:

