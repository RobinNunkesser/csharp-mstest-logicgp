using System.Text.Json.Serialization;

namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

[JsonDerivedType(typeof(ClassificationTrainingOptionV2))]
public interface ITrainingOption
{
    int TrainingTime { get; set; }

    int? Seed { get; set; }

    string? OutputFolder { get; set; }

    IValidationOption? ValidationOption { get; set; }
}