from flask import Flask, request
from flask import render_template, flash, redirect, url_for
from flask import jsonify
from flask_sqlalchemy import SQLAlchemy
import os
import sqlalchemy as sa
from flask_wtf.file import FileAllowed
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager
from flask_login import UserMixin
from flask_login import current_user, login_user
from flask_login import logout_user
from flask_login import login_required
from urllib.parse import urlparse
from flask_wtf import FlaskForm
from wtforms import PasswordField, BooleanField
from wtforms.validators import ValidationError, Email, EqualTo
from hashlib import md5
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
from wtforms import StringField, TextAreaField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired, Length
from flask_bootstrap import Bootstrap
import qrcode
from io import BytesIO
import base64
from datetime import datetime, timedelta
import secrets
from functools import wraps
from flask import abort
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from logging.handlers import RotatingFileHandler


log_handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=3)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logging.basicConfig(level=logging.INFO, handlers=[log_handler])

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
login = LoginManager(app)
login.login_view = 'login'
app.config['SECRET_KEY'] = 'you-will-never-guess'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or \
                                        'sqlite:///' + os.path.join(basedir, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
bootstrap = Bootstrap(app)

db = SQLAlchemy(app)
migrate = Migrate(app, db)


class AICategoryClassifier:
    def __init__(self):
        self.model = None
        self.category_embeddings = None
        self.categories = []
        self.logger = logging.getLogger(__name__)

    def initialize_model(self):
        try:
            self.model = SentenceTransformer(
                'sentence-transformers/paraphrase-albert-small-v2',
                device='cpu'
            )
            self.logger.info("ИИ-модель инициализирована")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации модели: {str(e)}")
            raise

    def train(self):
        with app.app_context():
            categories = EventCategory.query.all()
            if not categories:
                self.logger.warning("Нет категорий для обучения")
                return

            self.categories = [c.name for c in categories]

            descriptions = [
                "Выставки картин, скульптуры, инсталляции, перформансы, художественные мастер-классы, "
                "лекции по искусству, галереи, музеи, вернисажи, арт-объекты, живопись, графика, "
                "творческие мастерские, искусствоведение, художественные школы, арт-терапия, "
                "художественные конкурсы, арт-резиденции",

                "Ориентационные недели, дни первокурсников, студенческие клубы, инициативы, "
                "внеучебная деятельность, кампусные события, студенческие соревнования, "
                "студсовет, кураторство, тьюторство, студгородок, академические группы",

                "Ярмарки вакансий, карьерные консультации, собеседования, резюме, "
                "трудоустройство, профессиональное развитие, встречи с работодателями",

                "Лекции, семинары, научные конференции, учебные курсы, образование",

                "Фестивали музыки, кино, еды, культурные фестивали, концерты, вечеринки, "
                "развлекательные мероприятия, кинопоказы, конкурсы, шоу, развлекательные "
                "события массового характера, открытые площадки, музыкальные события, "
                "танцевальные мероприятия, праздничные события, городские праздники"
            ]

            self.category_embeddings = self.model.encode(descriptions)
            self.logger.info("Модель переобучена с четким разделением фестивалей")

    def predict(self, title, description):
        try:
            text = f"{title} {description}".lower()

            entertainment_keywords = {
                'фестиваль', 'концерт', 'вечеринк', 'развлечен', 'кинопоказ',
                'квиз', 'стендап', 'шоу', 'праздник', 'дискотек', 'танц', 'музык',
                'конкурс', 'выступлен', 'аттракцион', 'ярмарк', 'гулян', 'карнавал',
                'квест', 'ролева', 'игр', 'головоломк', 'загадк', 'викторин', 'тест-драйв',
                'тимбилдинг', 'мафия', 'крокодил', 'алиас', 'квн', 'игротека'
            }

            if any(keyword in text for keyword in entertainment_keywords):
                entertainment_cat = EventCategory.query.filter_by(name="Развлечения").first()
                if entertainment_cat:
                    self.logger.info(f"Определена категория 'Развлечения' по ключевым словам для: '{title}'")
                    return entertainment_cat

            student_keywords = {
                'студенч', 'первокурсник', 'факультет', 'куратор', 'студсовет',
                'адаптац', 'студвесн', 'университ', 'академ', 'групп', 'поток',
                'лекци', 'семинар', 'занят', 'учебн', 'курс', 'экзамен', 'зачет'
            }
            if any(keyword in text for keyword in student_keywords):
                student_cat = EventCategory.query.filter_by(name="Студенческая жизнь").first()
                if student_cat:
                    return student_cat

            if not self.model:
                self.initialize_model()
            if not hasattr(self, 'category_embeddings'):
                self.train()

            text_embedding = self.model.encode(text)
            similarity_scores = cosine_similarity(
                text_embedding.reshape(1, -1),
                self.category_embeddings
            )[0]

            best_idx = similarity_scores.argmax()

            if 'квест' in text and self.categories[best_idx] == "Образование":
                entertainment_cat = EventCategory.query.filter_by(name="Развлечения").first()
                if entertainment_cat:
                    self.logger.info(f"Переназначение квеста из 'Образование' в 'Развлечения': '{title}'")
                    return entertainment_cat

            self.logger.debug(
                f"Классификация: '{title}'\n"
                f"Категория: {self.categories[best_idx]}\n"
                f"Оценки: {dict(zip(self.categories, similarity_scores.round(2)))}"
            )

            return EventCategory.query.filter_by(name=self.categories[best_idx]).first()

        except Exception as e:
            self.logger.error(f"Ошибка классификации: {str(e)}")
            return EventCategory.query.first()

        except Exception as e:
            self.logger.error(f"Ошибка классификации: {str(e)}")
            return EventCategory.query.first()


ai_classifier = AICategoryClassifier()

@app.before_request
def init_ai_classifier():
    if not hasattr(app, 'ai_initialized'):
        with app.app_context():
            ai_classifier.initialize_model()
            ai_classifier.train()
        app.ai_initialized = True


class EventCategory(db.Model):
    __tablename__ = 'event_category'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.String(200))

    @staticmethod
    def get_default_categories():
        return [
            ("Искусство", "Выставки, мастер-классы, перформансы"),
            ("Студенческая жизнь", "Студенческие мероприятия, клубы, сообщества"),
            ("Образование", "Лекции, семинары, учебные программы"),
            ("Карьера", "Ярмарки вакансий, карьерные консультации"),
            ("Развлечения", "Концерты, вечеринки, культурные мероприятия")
        ]

    @staticmethod
    def update_categories():
        current_categories = {c.name for c in EventCategory.query.all()}
        default_categories = {name for name, desc in EventCategory.get_default_categories()}

        for name in current_categories - default_categories:
            EventCategory.query.filter_by(name=name).delete()

        for name, desc in EventCategory.get_default_categories():
            if name not in current_categories:
                db.session.add(EventCategory(name=name, description=desc))

        db.session.commit()

class EventAttendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    event_id = db.Column(db.Integer, db.ForeignKey('event.id'), nullable=False)
    attended = db.Column(db.Boolean, default=False)
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)
    marked_by_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    user = db.relationship('User', foreign_keys=[user_id], backref='attendances')
    event = db.relationship('Event', backref='attendances')
    marked_by = db.relationship('User', foreign_keys=[marked_by_id])

class AttendanceForm(FlaskForm):
    attended = BooleanField('Присутствовал на мероприятии')
    submit = SubmitField('Сохранить')

class UserListForm(FlaskForm):
    search = StringField('Поиск пользователей')
    submit = SubmitField('Найти')

class UserAttendanceForm(FlaskForm):
    event_id = SelectField('Мероприятие', coerce=int, validators=[DataRequired()])
    attended = BooleanField('Присутствовал')
    submit = SubmitField('Сохранить')

class Event(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    organizer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    category_id = db.Column(
        db.Integer,
        db.ForeignKey('event_category.id', name='fk_event_category_id'),  # Явное имя!
        nullable=True
    )
    category = db.relationship('EventCategory', backref='events')

    organizer = db.relationship('User', backref=db.backref('events', lazy=True))

class EventForm(FlaskForm):
    title = StringField('Название мероприятия', validators=[DataRequired()])
    description = TextAreaField('Описание', validators=[DataRequired()])
    date = StringField('Дата и время (ГГГГ-ММ-ДД ЧЧ:ММ)', validators=[DataRequired()])
    location = StringField('Место проведения', validators=[DataRequired()])
    submit = SubmitField('Создать мероприятие')

    def validate_date(self, field):
        try:
            datetime.strptime(field.data, '%Y-%m-%d %H:%M')
        except ValueError:
            raise ValidationError('Пожалуйста, введите дату в формате ГГГГ-ММ-ДД ЧЧ:ММ')

class UserRoles:
    ADMIN = 'admin'
    ORGANIZER = 'organizer'
    EXPERT = 'expert'
    PARTICIPANT = 'participant'

    @classmethod
    def all_roles(cls):
        return [cls.ADMIN, cls.ORGANIZER, cls.EXPERT, cls.PARTICIPANT]


def role_required(*roles):
    def wrapper(view_func):
        @wraps(view_func)
        def wrapped_view(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role not in roles:
                abort(403)
            return view_func(*args, **kwargs)

        return wrapped_view

    return wrapper


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500


class LoginForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired()])
    password = PasswordField('Пароль', validators=[DataRequired()])
    remember_me = BooleanField('Запомнить меня')
    submit = SubmitField('Войти')


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    about_me = db.Column(db.String(140))
    avatar = db.Column(db.String(120))
    qr_code_token = db.Column(db.String(32), index=True)
    qr_code_token_expiration = db.Column(db.DateTime)

    role = db.Column(db.String(20), default=UserRoles.PARTICIPANT)

    def can_be_viewed_by(self, viewer):
        return (
                viewer.is_admin or
                viewer.is_organizer or
                viewer.is_expert or
                viewer.id == self.id
        )

    def avatar_url(self, size=128):
        if self.avatar:
            return url_for('static', filename=f'uploads/avatars/{self.avatar}')
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return f'https://www.gravatar.com/avatar/{digest}?d=identicon&s={size}'

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_qr_token(self, expires_in=3600):
        self.qr_code_token = secrets.token_hex(16)
        self.qr_code_token_expiration = datetime.utcnow() + timedelta(seconds=expires_in)
        db.session.add(self)
        db.session.commit()
        return self.qr_code_token

    def revoke_qr_token(self):
        self.qr_code_token_expiration = datetime.utcnow() - timedelta(seconds=1)
        db.session.add(self)
        db.session.commit()

    def check_qr_token(self, token):
        if self.qr_code_token == token and self.qr_code_token_expiration > datetime.utcnow():
            return True
        return False

    def get_qr_code(self):
        if not self.qr_code_token or self.qr_code_token_expiration < datetime.utcnow():
            self.generate_qr_token()

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr_data = f"{self.id}:{self.qr_code_token}"
        qr.add_data(qr_data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    ratings_received = db.relationship('UserRating',
                                       foreign_keys='UserRating.rated_user_id',
                                       backref='rated_user',
                                       lazy='dynamic')

    ratings_given = db.relationship('UserRating',
                                    foreign_keys='UserRating.rater_id',
                                    backref='rater',
                                    lazy='dynamic')

    def get_average_rating(self):
        avg = db.session.query(db.func.avg(UserRating.rating)) \
            .filter(UserRating.rated_user_id == self.id) \
            .scalar()
        return round(avg, 1) if avg else 0.0

    def has_rated(self, user):
        return self.ratings_given.filter_by(rated_user_id=user.id).first() is not None

    @property
    def is_admin(self):
        return self.role == UserRoles.ADMIN

    @property
    def is_organizer(self):
        return self.role == UserRoles.ORGANIZER

    @property
    def is_expert(self):
        return self.role == UserRoles.EXPERT

    @property
    def is_participant(self):
        return self.role == UserRoles.PARTICIPANT

    @property
    def attended_events(self):
        return db.session.query(Event).join(
            EventAttendance,
            (EventAttendance.event_id == Event.id) &
            (EventAttendance.user_id == self.id)
        ).all()

    def get_attendance_status(self, event_id):
        attendance = EventAttendance.query.filter_by(
            event_id=event_id,
            user_id=self.id
        ).first()
        return attendance.attended if attendance else False


class UserRating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rater_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    rated_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('rater_id', 'rated_user_id', name='_user_rating_uc'),
    )


class RatingForm(FlaskForm):
    rating = SelectField('Оценка', choices=[(1, '1'), (2, '2'), (3, '3'), (4, '4'), (5, '5')],
                         coerce=int, validators=[DataRequired()])
    submit = SubmitField('Оценить')


@login.user_loader
def load_user(id):
    return db.session.get(User, int(id))


class RegistrationForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired()])
    email = StringField('Адрес электронной почты', validators=[DataRequired(), Email()])
    password = PasswordField('Придумайте пароль', validators=[DataRequired()])
    password2 = PasswordField(
        'Повторите пароль', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Подтвердить')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Это имя уже занято.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('К этому адресу электронной почты уже привязан аккаунт.')


class EditProfileForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired()])
    about_me = TextAreaField('Обо мне', validators=[Length(min=0, max=140)])
    avatar = FileField('Фото профиля', validators=[FileAllowed(['jpg', 'png', 'jpeg'])])
    remove_avatar = BooleanField('Удалить фото профиля')
    submit = SubmitField('Подтвердить изменения')

    def __init__(self, original_username, *args, **kwargs):
        super(EditProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username

    def validate_username(self, username):
        if username.data != self.original_username:
            user = User.query.filter_by(username=self.username.data).first()
            if user is not None:
                raise ValidationError('Это имя уже занято')


@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'UserRoles': UserRoles}


@app.route('/')
def index():
    categories = EventCategory.query.options(
        db.joinedload(EventCategory.events)
    ).order_by(EventCategory.name).all()

    uncategorized = [e for e in Event.query.all() if not e.category_id]
    if uncategorized:
        uncat_category = type('', (), {'name': 'Без категории', 'events': uncategorized})
        categories.append(uncat_category)

    return render_template('index.html', categories=categories)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = db.session.scalar(
            sa.select(User).where(User.username == form.username.data))
        if user is None or not user.check_password(form.password.data):
            flash('Неправильный логин или пароль')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        parsed_url = urlparse(next_page)
        if not next_page or parsed_url.netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Вход', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Вы зарегистрированы')
        return redirect(url_for('login'))
    return render_template('register.html', title='Регистрация', form=form)


@app.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    events = Event.query.order_by(Event.date.desc()).all()

    if not user.can_be_viewed_by(current_user):
        abort(403)

    return render_template('user.html',
                           user=user,
                           events=events)


@app.route('/change_role/<username>', methods=['POST'])
@login_required
@role_required(UserRoles.ADMIN)
def change_role(username):
    user = User.query.filter_by(username=username).first_or_404()
    new_role = request.form.get('new_role')

    if new_role not in UserRoles.all_roles():
        flash('Недопустимая роль', 'error')
        return redirect(url_for('user', username=username))

    user.role = new_role
    db.session.commit()
    flash(f'Роль пользователя {user.username} изменена на {new_role}', 'success')
    return redirect(url_for('user', username=username))


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    username = request.args.get('username', current_user.username)
    return redirect(url_for('edit_profile_with_username', username=username))


@app.route('/edit_profile/<username>', methods=['GET', 'POST'])
@login_required
def edit_profile_with_username(username):
    user = User.query.filter_by(username=username).first_or_404()
    is_owner = current_user.id == user.id

    qr_token = request.args.get('qr_token')
    has_qr_access = False

    if qr_token and user.check_qr_token(qr_token):
        has_qr_access = True
    elif not is_owner and not current_user.is_admin:
        flash('У вас нет прав для редактирования этого профиля', 'error')
        return redirect(url_for('index'))

    form = EditProfileForm(user.username)

    if form.validate_on_submit():
        if has_qr_access:
            if form.username.data != user.username:
                flash('Вы не можете изменять имя пользователя с временным доступом', 'error')
                return redirect(url_for('edit_profile_with_username', username=user.username, qr_token=qr_token))
        else:
            user.username = form.username.data

        user.about_me = form.about_me.data

        if form.remove_avatar.data and user.avatar:
            avatar_path = os.path.join(basedir, 'static', 'uploads', 'avatars')
            old_avatar = os.path.join(avatar_path, user.avatar)
            if os.path.exists(old_avatar):
                os.remove(old_avatar)
            user.avatar = None
        elif form.avatar.data:
            avatar = form.avatar.data
            filename = secure_filename(f"{user.id}_{avatar.filename}")
            avatar_path = os.path.join(basedir, 'static', 'uploads', 'avatars')
            os.makedirs(avatar_path, exist_ok=True)
            avatar.save(os.path.join(avatar_path, filename))
            if user.avatar:
                old_avatar = os.path.join(avatar_path, user.avatar)
                if os.path.exists(old_avatar):
                    os.remove(old_avatar)
            user.avatar = filename

        db.session.commit()
        flash('Изменения успешно сохранены')

        if has_qr_access:
            return redirect(url_for('edit_profile_with_username', username=user.username, qr_token=qr_token))
        return redirect(url_for('edit_profile_with_username', username=user.username))

    elif request.method == 'GET':
        form.username.data = user.username
        form.about_me.data = user.about_me

    return render_template(
        'edit_profile.html',
        title='Редактировать профиль',
        form=form,
        user=user,
        is_owner=is_owner,
        has_qr_access=has_qr_access
    )


@app.route('/generate_qr')
@login_required
def generate_qr():
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(current_user.username)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return render_template('qr_code.html',
                         qr_code=img_str,
                         username=current_user.username)


@app.route('/scan_qr', methods=['GET', 'POST'])
@login_required
def scan_qr():
    if request.method == 'POST':
        qr_data = request.form.get('qr_data')
        if not qr_data:
            flash('Не удалось прочитать QR-код')
            return redirect(url_for('scan_qr'))

        try:
            username = qr_data.strip()
            user = User.query.filter_by(username=username).first()

            if user:
                return redirect(url_for('user', username=user.username))
            else:
                flash('Пользователь не найден')
        except Exception as e:
            flash('Неверный формат QR-кода')

        return redirect(url_for('scan_qr'))

    return render_template('scan_qr.html', title='Сканировать QR-код')


@app.route('/rate_user/<username>', methods=['GET', 'POST'])
@login_required
def rate_user(username):
    user_to_rate = User.query.filter_by(username=username).first_or_404()

    if current_user.id == user_to_rate.id:
        flash('Вы не можете оценивать себя', 'error')
        return redirect(url_for('user', username=username))

    if not (current_user.is_admin or current_user.is_expert):
        flash('Только администраторы и эксперты могут оценивать пользователей', 'error')
        return redirect(url_for('user', username=username))

    form = RatingForm()

    existing_rating = UserRating.query.filter_by(
        rater_id=current_user.id,
        rated_user_id=user_to_rate.id
    ).first()

    if form.validate_on_submit():
        if existing_rating:
            existing_rating.rating = form.rating.data
            flash('Оценка обновлена', 'success')
        else:
            new_rating = UserRating(
                rater_id=current_user.id,
                rated_user_id=user_to_rate.id,
                rating=form.rating.data
            )
            db.session.add(new_rating)
            flash('Спасибо за оценку!', 'success')

        db.session.commit()
        return redirect(url_for('user', username=username))

    if existing_rating:
        form.rating.data = existing_rating.rating

    return render_template('rate_user.html',
                           title=f'Оценить {user_to_rate.username}',
                           user=user_to_rate,
                           form=form)


def create_first_admin():
    with app.app_context():
        try:
            admin = User.query.filter_by(email='admin@example.com').first()
            if not admin:
                admin = User(
                    username='admin',
                    email='admin@example.com',
                    role=UserRoles.ADMIN
                )
                admin.set_password('admin123')
                db.session.add(admin)
                print("Администратор создан")
            else:
                if admin.role != UserRoles.ADMIN:
                    admin.role = UserRoles.ADMIN
                    print("Роль администратора обновлена")
                else:
                    print("Администратор уже существует с правильной ролью")

            organizer = User.query.filter_by(email='org@example.com').first()
            if not organizer:
                organizer = User(
                    username='org',
                    email='org@example.com',
                    role=UserRoles.ORGANIZER
                )
                organizer.set_password('org123')
                db.session.add(organizer)
                print("Организатор создан")
            else:
                if organizer.role != UserRoles.ORGANIZER:
                    organizer.role = UserRoles.ORGANIZER
                    print("Роль организатора обновлена")
                else:
                    print("Организатор уже существует с правильной ролью")

            expert = User.query.filter_by(email='expert@example.com').first()
            if not expert:
                expert = User(
                    username='expert',
                    email='expert@example.com',
                    role=UserRoles.EXPERT
                )
                expert.set_password('expert123')
                db.session.add(expert)
                print("Эксперт создан")
            else:
                if expert.role != UserRoles.EXPERT:
                    expert.role = UserRoles.EXPERT
                    print("Роль эксперта обновлена")
                else:
                    print("Эксперт уже существует с правильной ролью")
            if not User.query.filter_by(email='participant1@example.com').first():
                participant = User(
                    username='participant1',
                    email='participant1@example.com',
                    role=UserRoles.PARTICIPANT
                )
                participant.set_password('part123')
                db.session.add(participant)
                print("Тестовый участник создан")
            if not EventCategory.query.first():
                for name, desc in EventCategory.get_default_categories():
                    db.session.add(EventCategory(name=name, description=desc))
                db.session.commit()

            EventCategory.update_categories()

            db.session.commit()
            print("Все изменения успешно сохранены в бд")

        except Exception as e:
            db.session.rollback()
            print(f"❌ Ошибка при создании пользователей: {str(e)}")
            raise


def initialize_ai():
    with app.app_context():
        try:
            ai_classifier.initialize_model()
            ai_classifier.train()
            app.logger.info("ИИ-классификатор успешно инициализирован")
        except Exception as e:
            app.logger.error(f"Ошибка инициализации ИИ: {str(e)}")

@app.before_request
def before_first_request():
    if not hasattr(app, 'ai_initialized'):
        initialize_ai()
        app.ai_initialized = True


@app.route('/create_event', methods=['GET', 'POST'])
@login_required
def create_event():
    form = EventForm()
    if form.validate_on_submit():
        try:
            category = ai_classifier.predict(form.title.data, form.description.data)

            if not category:
                category = EventCategory.query.first()
                flash("Не удалось определить категорию", 'warning')
            else:
                flash(f"Категория определена как: {category.name}", 'info')

            event = Event(
                title=form.title.data,
                description=form.description.data,
                date=datetime.strptime(form.date.data, '%Y-%m-%d %H:%M'),
                location=form.location.data,
                organizer_id=current_user.id,
                category_id=category.id
            )

            db.session.add(event)
            db.session.commit()
            flash('Мероприятие успешно создано!', 'success')
            return redirect(url_for('index'))

        except Exception as e:
            flash(f'Ошибка при создании мероприятия: {str(e)}', 'error')
            db.session.rollback()

    return render_template('create_event.html', form=form)


@app.route('/users')
@login_required
@role_required(UserRoles.ADMIN, UserRoles.ORGANIZER, UserRoles.EXPERT)
def user_list():
    search_query = request.args.get('search', '').strip()
    query = User.query.order_by(User.username)

    if search_query:
        query = query.filter(User.username.ilike(f'%{search_query}%'))

    users = query.all()
    events = Event.query.order_by(Event.date.desc()).all()

    return render_template('user_list.html',
                         users=users,
                         events=events)


@app.route('/mark_attendance_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
@role_required(UserRoles.ADMIN, UserRoles.ORGANIZER)
def mark_attendance_user(user_id):
    user = User.query.get_or_404(user_id)
    form = UserAttendanceForm()

    form.event_id.choices = [(e.id, e.title) for e in Event.query.order_by(Event.date.desc()).all()]

    if form.validate_on_submit():
        attendance = EventAttendance.query.filter_by(
            event_id=form.event_id.data,
            user_id=user_id
        ).first()

        if not attendance:
            attendance = EventAttendance(
                event_id=form.event_id.data,
                user_id=user_id,
                marked_by_id=current_user.id
            )
            db.session.add(attendance)

        attendance.attended = form.attended.data
        db.session.commit()
        flash('Посещение обновлено', 'success')
        return redirect(url_for('user', username=user.username))

    return render_template('mark_attendance_user.html', form=form, user=user)

@app.route('/mark_attendance/<int:event_id>/<int:user_id>', methods=['GET', 'POST'])
@login_required
@role_required(UserRoles.ADMIN, UserRoles.ORGANIZER, UserRoles.EXPERT)
def mark_attendance(event_id, user_id):
    event = Event.query.get_or_404(event_id)
    user = User.query.get_or_404(user_id)
    attendance = EventAttendance.query.filter_by(event_id=event_id, user_id=user_id).first()

    if not attendance:
        attendance = EventAttendance(event_id=event_id, user_id=user_id)
        db.session.add(attendance)

    form = AttendanceForm()

    if form.validate_on_submit():
        attendance.attended = form.attended.data
        db.session.commit()
        flash('Посещение обновлено', 'success')
        return redirect(url_for('user', username=user.username))

    form.attended.data = attendance.attended if attendance else False
    return render_template('mark_attendance.html', form=form, user=user, event=event)


@app.route('/event_attendance/<int:event_id>')
@login_required
@role_required(UserRoles.ADMIN, UserRoles.ORGANIZER)
def event_attendance(event_id):
    event = Event.query.get_or_404(event_id)

    all_users = all_users = User.query.order_by(User.username).all()

    attendance_records = EventAttendance.query.filter_by(
        event_id=event_id
    ).all()

    users_data = []
    for user in all_users:
        attendance = next(
            (a for a in attendance_records if a.user_id == user.id),
            None
        )

        users_data.append({
            'id': user.id,
            'username': user.username,
            'avatar_url': user.avatar_url(40),
            'is_attended': attendance.attended if attendance else False,
            'attendance_id': attendance.id if attendance else None
        })

    print(f"Все пользователи в системе: {len(all_users)}")
    print(f"Записи о посещаемости: {len(attendance_records)}")
    print("Детали:")
    for user in users_data:
        print(f"ID: {user['id']}, Username: {user['username']}, Attended: {user['is_attended']}")

    return render_template(
        'event_attendance.html',
        event=event,
        users=users_data,
        attendees_count=len([u for u in users_data if u['is_attended']])
    )


@app.route('/toggle_attendance/<int:event_id>/<int:user_id>', methods=['POST'])
@login_required
@role_required(UserRoles.ADMIN, UserRoles.ORGANIZER)
def toggle_attendance(event_id, user_id):
    try:
        attendance = EventAttendance.query.filter_by(
            event_id=event_id,
            user_id=user_id
        ).first()

        if not attendance:
            attendance = EventAttendance(
                event_id=event_id,
                user_id=user_id,
                attended=True,
                marked_by_id=current_user.id
            )
            db.session.add(attendance)
            new_status = True
        else:
            new_status = not attendance.attended
            attendance.attended = new_status
            attendance.marked_by_id = current_user.id

        db.session.commit()

        attendees_count = EventAttendance.query.filter_by(
            event_id=event_id,
            attended=True
        ).count()

        return jsonify({
            'success': True,
            'attended': new_status,
            'attendance_id': attendance.id,
            'new_count': attendees_count
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/bulk_rate', methods=['POST'])
@login_required
@role_required(UserRoles.ADMIN, UserRoles.EXPERT)
def bulk_rate():
    data = request.get_json()
    user_ids = data.get('user_ids')
    rating = data.get('rating')

    if not user_ids or not rating:
        return jsonify({'success': False, 'error': 'Неверные данные'})

    try:
        for user_id in user_ids:
            existing = UserRating.query.filter_by(
                rater_id=current_user.id,
                rated_user_id=user_id
            ).first()

            if existing:
                existing.rating = rating
            else:
                new_rating = UserRating(
                    rater_id=current_user.id,
                    rated_user_id=user_id,
                    rating=rating
                )
                db.session.add(new_rating)

        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/bulk_attendance', methods=['POST'])
@login_required
@role_required(UserRoles.ADMIN, UserRoles.ORGANIZER)
def bulk_attendance():
    data = request.get_json()
    user_ids = data.get('user_ids')
    event_id = data.get('event_id')
    attended = data.get('attended', True)

    if not user_ids or not event_id:
        return jsonify({'success': False, 'error': 'Неверные данные'})

    try:
        for user_id in user_ids:
            attendance = EventAttendance.query.filter_by(
                event_id=event_id,
                user_id=user_id
            ).first()

            if attendance:
                attendance.attended = attended
                attendance.marked_by_id = current_user.id
            else:
                new_attendance = EventAttendance(
                    event_id=event_id,
                    user_id=user_id,
                    attended=attended,
                    marked_by_id=current_user.id
                )
                db.session.add(new_attendance)

        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/delete_event/<int:event_id>', methods=['POST'])
@login_required
@role_required(UserRoles.ADMIN)
def delete_event(event_id):
    event = Event.query.get_or_404(event_id)

    EventAttendance.query.filter_by(event_id=event_id).delete()

    db.session.delete(event)
    db.session.commit()

    flash('Мероприятие успешно удалено', 'success')
    return redirect(url_for('index'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_first_admin()
    print("\n * сервер запущен! перейдите по ссылке: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)