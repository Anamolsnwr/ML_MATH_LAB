from flask import Flask, render_template
from routes.math_routes import math_bp
from routes.ml_routes import ml_bp

app = Flask(__name__)

# Register separated blueprints: Math Calculation Modules & ML Algorithms
app.register_blueprint(math_bp)
app.register_blueprint(ml_bp)

# Enable both bare endpoints (e.g., 'identity_page') and blueprint endpoints (e.g., 'math.identity_page')
for rule in list(app.url_map.iter_rules()):
    if '.' in rule.endpoint:
        bare = rule.endpoint.split('.', 1)[1]
        if bare not in app.view_functions:
            app.add_url_rule(
                rule.rule,
                endpoint=bare,
                view_func=app.view_functions[rule.endpoint],
                methods=list(rule.methods)
            )


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)