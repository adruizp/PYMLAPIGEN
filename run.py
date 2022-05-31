import pymlapigen
import sys
import os


if (__name__ == "__main__"):
    pymlapigen.flask_app.config['APP_FOLDER'] = os.path.join(os.getcwd(), "pymlapigen")
    if len(sys.argv) == 1:
        pymlapigen.flask_app.run(debug=True)   
    # Custom ip
    elif len(sys.argv) == 2:
        pymlapigen.flask_app.run(debug=True, host=sys.argv[1])
    # Custom ip and port
    elif len(sys.argv) == 3:
        pymlapigen.flask_app.run(debug=True, host=sys.argv[1],port=sys.argv[2])