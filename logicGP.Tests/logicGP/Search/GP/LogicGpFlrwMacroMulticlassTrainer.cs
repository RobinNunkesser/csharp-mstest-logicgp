using Italbytz.Adapters.Algorithms.AI.Search.GP.Control;
using Microsoft.ML;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP;

/// <summary>
///     ///     A class representing a trainer for the LogicGP algorithm.
///     It is used to train the LogicGP algorithm for multiclass classification
///     problems.
///     The trainer uses the LogicGpFlrw algorithm and the macro accuracy metric.
///     It extends the LogicGpTrainerBase class and provides methods for creating a
///     transformer and parameterizing the algorithm.
///     The trainer is used to train the LogicGP algorithm on a given dataset and
///     generate a transformer that can be used for predictions.
/// </summary>
public class LogicGpFlrwMacroMulticlassTrainer(
    LogicGpAlgorithm algorithm,
    DataManager data)
    : LogicGpTrainerBase<ITransformer>(algorithm, data)
{
    protected override void ParameterizeAlgorithm(
        LogicGpAlgorithm logicGpAlgorithm)
    {
        logicGpAlgorithm.UseFullInitialization = true;
        logicGpAlgorithm.UsedWeighting = LogicGpAlgorithm.Weighting.Computed;
        logicGpAlgorithm.WeightMutationToUse =
            LogicGpAlgorithm.WeightMutation.None;
        logicGpAlgorithm.UsedAccuracy = LogicGpAlgorithm.Accuracies.Macro;
    }
}