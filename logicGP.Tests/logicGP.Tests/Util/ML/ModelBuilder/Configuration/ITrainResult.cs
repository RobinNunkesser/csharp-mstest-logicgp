namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

public interface ITrainResult
{
    IEnumerable<ITrial>? Trials { get; set; }

    string? PipelineSchema { get; set; }

    Dictionary<string, EstimatorType>? Estimators { get; set; }

    string? MetricName { get; set; }

    string? ModelFilePath { get; set; }
}