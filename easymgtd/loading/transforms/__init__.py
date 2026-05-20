"""
Transform registration module.

Importing this package triggers all transform registrations via side effects.
Each transform file registers itself with DatasetRegistry upon import.
"""

# Import all transform modules to trigger @DatasetRegistry.register decorators
from . import truthfulqa
from . import squad1
from . import narrativeqa
from . import aitextdetect_binary
from . import aitextdetect_attribution
from . import aitextdetect_incremental
