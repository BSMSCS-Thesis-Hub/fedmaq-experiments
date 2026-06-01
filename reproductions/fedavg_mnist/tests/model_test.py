"""Used to test the model and the data partitionning."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import model


def test_cnn_size_mnist() -> None:
    """Test number of parameters with MNIST-sized inputs."""
    # Prepare
    net = model.Net()
    expected = 1_663_370

    # Execute
    actual = sum([p.numel() for p in net.parameters()])

    # Assert
    assert actual == expected
