namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

public interface ITrial
{
    Parameter? Parameter { get; set; }

    string? TrainerName { get; set; }

    double Score { get; set; }

    double RuntimeInSeconds { get; set; }
}