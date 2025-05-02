namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

public class ClassificationTrainingOptionV2 : MBConfig, ITrainingOption
{
    public override int Version => 2;

    public override string? Type => "ClassificationTrainingOption";
    public bool Subsampling { get; set; }
    public string? LabelColumn { get; set; }
    public string[]? AvailableTrainers { get; set; }
    public int TrainingTime { get; set; }
    public int? Seed { get; set; }
    public string? OutputFolder { get; set; }
    public IValidationOption? ValidationOption { get; set; }
}