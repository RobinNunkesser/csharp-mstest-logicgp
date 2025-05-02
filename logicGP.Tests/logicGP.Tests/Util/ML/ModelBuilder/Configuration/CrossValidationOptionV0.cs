namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

public class CrossValidationOptionV0 : MBConfig, IValidationOption
{
    public override int Version => 0;

    public override string Type => "CrossValidationValidationOption";

    public int? NumberOfFolds { get; set; }
}