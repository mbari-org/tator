# #region agent log
import sys
try:
    from .middleware import KeycloakMiddleware
except Exception as e:
    import traceback
    sys.stderr.write("DEBUG tator_online.__init__ exception: " + type(e).__name__ + ": " + str(e) + "\n")
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    raise
# #endregion
