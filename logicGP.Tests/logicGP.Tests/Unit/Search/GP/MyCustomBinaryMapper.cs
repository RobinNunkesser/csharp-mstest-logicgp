using Italbytz.ML;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP;

public class MyCustomBinaryMapper
{
    public static Action<TSrc, TDst> GetMapping<TSrc, TDst>()
        where TSrc : class, new() where TDst : class, new()
    {
        return Fit;
    }

    private static void Fit<TSrc, TDst>(TSrc input, TDst output)
        where TSrc : class, new() where TDst : class, new()
    {
        if (output is not BinaryClassificationOutputSchema schema) return;
        schema.Score = 1.0f;
        schema.Probability = 1.0f;
        schema.PredictedLabel = 1;
    }
}