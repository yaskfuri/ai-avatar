from transformers import pipeline

# Load the sentiment-analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# Example sentences to analyze
sentences = [
    "She has a radiant smile and beautiful eyes.",
    "Her appearance is natural and unique.",
    "He has a very slender build and stylish looks.",
    "The person's physical traits are average and unremarkable.",
    "She exudes a confident and genuine charm.",
    "Her presence is captivating and elegant."
]

# Analyze and print sentiment for each sentence
for sentence in sentences:
    result = sentiment_analysis(sentence)[0]  # Get the first result
    label = result['label']                    # Sentiment label (e.g., POSITIVE, NEGATIVE)
    score = result['score']                    # Confidence score
    
    # Print the results
    print(f"Sentence: '{sentence}'")
    print(f"  Sentiment: {label} (Score: {score:.2f})")
    print()