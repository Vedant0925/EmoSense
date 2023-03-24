from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class MoodForm(FlaskForm):
    mood = StringField('Enter your mood:', validators=[DataRequired()])
    submit = SubmitField('Get Recommendations')
