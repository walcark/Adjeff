import numpy as np

from adjeff.utils import adjeff_logging, LoggerConfig, fft_convolve_2D
from adjeff.atmosphere import AtmosphereConfig, SunsatFactory


def main():
    res = fft_convolve_2D(
        np.random.rand(5000, 5000),
        np.random.rand(5000, 5000),
        padding="reflect",
        device="cpu"
    )

    config: AtmosphereConfig = AtmosphereConfig(
        aot=0.5,
        rh=0.0,
    )
    print(config)
    
    ss: SunsatFactory = SunsatFactory(config)

    print(ss.sat_le)
    print(ss.sun_le)
    print(ss.sat_sensor)
    print(ss.satellite_relative_position)


if __name__ == "__main__":
    main()
