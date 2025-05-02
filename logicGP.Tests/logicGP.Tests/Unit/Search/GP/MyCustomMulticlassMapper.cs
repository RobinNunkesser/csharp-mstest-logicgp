using Italbytz.ML;
using Microsoft.ML.Data;

namespace logicGP.Tests.Unit.Search.GP;

public class MyCustomMulticlassMapper
{
    public static Action<TSrc, TDst> GetMapping<TSrc, TDst>()
        where TSrc : class, new() where TDst : class, new()
    {
        return Fit;
    }

    private static void Fit<TSrc, TDst>(TSrc input, TDst output)
        where TSrc : class, new() where TDst : class, new()
    {
        if (output is not TernaryClassificationClassificationOutputSchema
            schema)
            return;
        schema.Score = new VBuffer<float>(3, [0.0f, 1.0f, 2.0f]);
        schema.Probability = new VBuffer<float>(3, [0.0f, 1.0f, 0.0f]);
        schema.PredictedLabel = 2;
    }
}