import pymlapigen
import sys
import os

def cli():
    # Loads configuration 
    pymlapigen.flask_app.config['APP_FOLDER'] = os.path.dirname(pymlapigen.__file__)

    # Check if user did input hosting IP
    if len(sys.argv) == 1:
        pymlapigen.flask_app.run()
    else:
        pymlapigen.flask_app.run(host=sys.argv[1])