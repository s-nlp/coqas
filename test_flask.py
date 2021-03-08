from flask import Flask, request
from my_functions import do_sum, answerer
app = Flask(__name__)

#@app.route("/")
#def hello():
    #return "Hello World2!"

@app.route("/", methods=["GET", "POST"])
def adder_page():
    errors = ""
    if request.method == "POST":
        request_string = None
        try:
            request_string = request.form["question"]
        except:
            errors += "<p>{!r} is not a string.</p>\n".format(request.form["question"])
        if request_string is not None:
            a = "blablalba"
            
            result = answerer(request_string)
            print ("RESULT", result)
            return '''
                <html>
                    <body>
                        <head>
                        <link rel="stylesheet" href='/static/style1.css' />
                        </head>
                        <p><style="height: 100px; width: 250px; font-size: 60%>{result}</p>
                        <p><a href="/">Click here too calculate again</a>
                    </body>
                </html>
            '''.format(result=result)

    return '''
        <html>
            <head>
            <link rel="stylesheet" href='/static/style.css' />
            </head>
            <body bgcolor="#E6E6FA">
                {errors}
                <p>Enter comparative question:</p>
                <form method="post" action=".">
                    <p><input name="question" value = "What is better for health tea or coffee?" style="height: 30px; width: 450px; font-size: 60%; font-family: verdana;"/></p>
                    <p><input type="submit" value="Answer" style="height: 50px; width: 250px; font-size: 60%; font-family: verdana;"/></p>
                </form>               
            </body>
        </html>
    '''.format(errors=errors)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port='6006')
    
    