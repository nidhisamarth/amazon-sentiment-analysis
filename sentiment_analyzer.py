import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import sys

class AmazonSentimentAnalyzer:
    """
    A production-ready sentiment analysis pipeline for Amazon product reviews
    """
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.id2label = {0: "Negative", 1: "Positive"}  # Binary classification
        self.model.eval()  # Set model to evaluation mode

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"

    def preprocess(self, text):
        """Clean and prepare text for the model"""
        if isinstance(text, str):
            return text.strip().replace("\n", " ")
        elif isinstance(text, list):
            return [t.strip().replace("\n", " ") for t in text]
        else:
            raise ValueError("Input must be a string or list of strings")

    def predict(self, text, include_probabilities=False):
        """
        Analyze sentiment in product reviews

        Args:
            text: A single review text or list of reviews
            include_probabilities: Whether to include class probabilities

        Returns:
            Sentiment analysis results including sentiment label and confidence
        """
        # Handle single text vs list input
        is_single_text = isinstance(text, str)
        if is_single_text:
            text = [text]

        # Preprocess the text
        processed_text = self.preprocess(text)

        # Tokenize input
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        # Convert to results
        results = []
        for i, (review, pred, probs) in enumerate(zip(processed_text, predictions, probabilities)):
            pred_label = self.id2label[pred.item()]
            confidence = probs[pred].item()

            result = {
                "text": review,
                "sentiment": pred_label,
                "confidence": confidence
            }

            if include_probabilities:
                result["probabilities"] = {
                    self.id2label[j]: prob.item()
                    for j, prob in enumerate(probs)
                }

            results.append(result)

        return results[0] if is_single_text else results

    def batch_analyze_file(self, file_path, output_path=None, batch_size=32):
        """
        Process reviews from a file in an efficient batch manner

        Args:
            file_path: Path to file with one review per line
            output_path: Where to save results (if None, returns results)
            batch_size: Number of reviews to process at once

        Returns:
            Analysis results if output_path is None
        """
        # Read reviews from file
        with open(file_path, 'r', encoding='utf-8') as f:
            reviews = [line.strip() for line in f if line.strip()]

        results = []
        # Process in batches
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i+batch_size]
            batch_results = self.predict(batch)
            results.extend(batch_results)

        # Save to file if requested
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"{result['sentiment']}\t{result['confidence']:.4f}\t{result['text']}\n")
            return f"Results saved to {output_path}"

        return results


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Analyze sentiment in Amazon product reviews")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument("--input", type=str, required=True,
                        help="Input text or file path (use --file flag for file)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (only used with --file flag)")
    parser.add_argument("--file", action="store_true",
                        help="Treat input as a file path with one review per line")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing (only used with --file flag)")
    parser.add_argument("--probabilities", action="store_true",
                        help="Include class probabilities in output")

    args = parser.parse_args()

    try:
        # Initialize the sentiment analyzer
        analyzer = AmazonSentimentAnalyzer(args.model_path)

        if args.file:
            # Process file
            result = analyzer.batch_analyze_file(
                args.input,
                args.output,
                args.batch_size
            )
            if args.output:
                print(result)
            else:
                # Print results to console
                for item in result:
                    print(f"{item['sentiment']}\t{item['confidence']:.4f}\t{item['text'][:100]}...")
        else:
            # Process single review
            result = analyzer.predict(args.input, include_probabilities=args.probabilities)
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.4f}")
            if args.probabilities:
                print("Class probabilities:")
                for sentiment, score in result['probabilities'].items():
                    print(f"  - {sentiment}: {score:.4f}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
