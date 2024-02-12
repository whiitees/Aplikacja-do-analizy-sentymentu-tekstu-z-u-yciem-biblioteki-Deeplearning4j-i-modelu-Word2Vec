import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

public class SentimentAnalysis {
    public static void main(String[] args) throws IOException {
        // Wczytanie modelu Word2Vec
        File modelFile = new ClassPathResource("path/to/word2vec/model.txt").getFile();
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(modelFile);

        // Przykładowe zdanie
        String sentence = "This is a great product!";
        
        // Tokenizacja tekstu
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new MyPreprocessor());
        List<String> tokens = tokenizerFactory.create(sentence).getTokens();

        // Obliczenie średniego wektora dla zdania
        INDArray vector = wordVectors.getWordVectorsMean(tokens);

        // Przewidywanie sentymentu
        double[] scores = classifier.predict(vector);
        double positiveScore = scores[1];
        double negativeScore = scores[0];

        if (positiveScore > negativeScore) {
            System.out.println("Positive sentiment");
        } else {
            System.out.println("Negative sentiment");
        }
    }
}
