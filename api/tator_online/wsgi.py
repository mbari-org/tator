"""
WSGI config for tator_online project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.1/howto/deployment/wsgi/
"""

import os
import sys
import traceback
import time
import signal

# #region agent log
def _load_wsgi():
    from django.core.wsgi import get_wsgi_application
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tator_online.settings")
    return get_wsgi_application()
# #endregion

try:
    application = _load_wsgi()
except Exception as e:
    import traceback as tb
    sys.stderr.write("DEBUG wsgi load exception: " + type(e).__name__ + ": " + str(e) + "\n")
    tb.print_exc(file=sys.stderr)
    sys.stderr.flush()
    raise
