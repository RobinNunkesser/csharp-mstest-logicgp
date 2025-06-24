using Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;
using Italbytz.AI.Search.GP.Individuals;
using Italbytz.ML;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP;

public class LogicGpMapping(IIndividual chosenIndividual)
{
    public Action<TSrc, TDst> GetMapping<TSrc, TDst>()
        where TSrc : class, new() where TDst : class, new()
    {
        return Map<TSrc, TDst>;
    }


    private void Map<TSrc, TDst>(TSrc arg1, TDst arg2)
        where TSrc : class, new() where TDst : class, new()
    {
        if (chosenIndividual is not Individual logicGpIndividual) return;
        if (logicGpIndividual.Genotype is not LogicGpGenotype logicGpGenotype)
            return;
        var gen = chosenIndividual.Genotype;
        var dst = ((LogicGpGenotype)gen).Predict<TSrc, TDst>(arg1);
        if (arg2 is IBinaryClassificationOutput
                binaryDestinationSchema &&
            dst is IBinaryClassificationOutput
                binaryPrediction)
        {
            binaryDestinationSchema.Probability =
                binaryPrediction.Probability;
            binaryDestinationSchema.Score =
                binaryPrediction.Score;
            binaryDestinationSchema.PredictedLabel =
                binaryPrediction.PredictedLabel;
        }
        else if
            (arg2 is IMulticlassClassificationOutput
                 multiDestinationSchema &&
             dst is IMulticlassClassificationOutput
                 multiPrediction)
        {
            multiDestinationSchema.Probability =
                multiPrediction.Probability;
            multiDestinationSchema.Score =
                multiPrediction.Score;
            multiDestinationSchema.PredictedLabel =
                multiPrediction.PredictedLabel;
        }
    }
}