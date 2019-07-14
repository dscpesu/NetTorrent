import flask
import subprocess

app = flask.Flask(__name__)

path_to_run = './' 
py_name = 'node2.py'  
args = ["python3", "{}{}".format(path_to_run, py_name)]

lrw=None


@app.route('/api/worker/start', methods = ['GET'])
def start():
    global lrw
    global output
    if lrw is not None:
        return flask.Response(status=409)  
    else: 
        lrw=subprocess.Popen(args)
        return flask.Response(status=202)
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=4500)