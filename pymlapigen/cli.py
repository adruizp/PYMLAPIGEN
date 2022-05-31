import pymlapigen
import sys
import os

def cli():
    # Loads configuration 
    pymlapigen.flask_app.config['APP_FOLDER'] = os.path.dirname(pymlapigen.__file__)

    # Default ip and port
    if len(sys.argv) == 1:
        pymlapigen.flask_app.run()
    # Custom ip
    elif len(sys.argv) == 2:
        pymlapigen.flask_app.run(host=sys.argv[1])
    # Custom ip and port
    elif len(sys.argv) == 3:
        pymlapigen.flask_app.run(host=sys.argv[1],port=sys.argv[2])