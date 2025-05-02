using System.Text.Json.Serialization;

namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

public class TrainingConfiguration : MBConfig, ITrainingConfiguration
{
    public override int Version => 5;
    public override string Type => "TrainingConfig";
    public virtual ScenarioType Scenario { get; set; }

    public virtual IDataSource? DataSource { get; set; }

    public virtual IEnvironment? Environment { get; set; }

    [JsonIgnore]
    public virtual AutoMLType? AutoMLType { get; set; } =
        Configuration.AutoMLType.Octopus;

    [JsonIgnore] public virtual ITrainResult? TrainResult { get; set; }

    public virtual ITrainingOption? TrainingOption { get; set; }

    [JsonIgnore]
    public virtual string? TrainingConfigurationFolder { get; set; }
}