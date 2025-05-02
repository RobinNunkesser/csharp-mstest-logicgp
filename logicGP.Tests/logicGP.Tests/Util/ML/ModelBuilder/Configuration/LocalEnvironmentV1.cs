using System.Text.Json.Serialization;

namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

public class LocalEnvironmentV1 : MBConfig, IEnvironment
{
    public override int Version => 1;
    public override string? Type { get; set; }
    [JsonIgnore] public EnvironmentType EnvironmentType { get; set; }
}