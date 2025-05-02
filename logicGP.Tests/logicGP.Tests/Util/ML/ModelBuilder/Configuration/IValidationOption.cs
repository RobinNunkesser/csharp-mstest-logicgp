using System.Text.Json.Serialization;

namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

[JsonDerivedType(typeof(FileValidationOptionV0))]
[JsonDerivedType(typeof(CrossValidationOptionV0))]
[JsonDerivedType(typeof(TrainValidationSplitOptionV0))]
public interface IValidationOption
{
}