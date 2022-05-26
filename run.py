import pymlapigen
import sys
import os


if (__name__ == "__main__"):
    pymlapigen.flask_app.config['APP_FOLDER'] = os.path.join(os.getcwd(), "pymlapigen")
    if len(sys.argv) == 1:
        pymlapigen.flask_app.run(debug=True)
    else:
        pymlapigen.flask_app.run(host=sys.argv[1])