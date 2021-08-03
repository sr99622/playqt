#include "alarmpanel.h"
#include "mainwindow.h"

AlarmPanel::AlarmPanel(QMainWindow *parent, int obj_id) : Panel(parent)
{
    this->obj_id = obj_id;

    minLimit = new AlarmNumberBox(key() + "/minLimit", MW->settings, "");
    maxLimit = new AlarmNumberBox(key() + "/maxLimit", MW->settings, "");
    minLimitTime = new AlarmNumberBox(key() + "/minLimitTime", MW->settings, "");
    maxLimitTime = new AlarmNumberBox(key() + "/maxLimitTime", MW->settings, "");

    int boxWidth = minLimit->fontMetrics().boundingRect("000000").width();
    minLimit->setMaximumWidth(boxWidth);
    maxLimit->setMaximumWidth(boxWidth);
    minLimitTime->setMaximumWidth(boxWidth);
    maxLimitTime->setMaximumWidth(boxWidth);

    chkMin = new AlarmCheckBox("Alarm if count goes below: ", key() + "/chkMin", MW->settings, false);
    chkMax = new AlarmCheckBox("Alarm if count goes above: ", key() + "/chkMax", MW->settings, false);

    QLabel *lbl00 = new QLabel(" for ");
    QLabel *lbl01 = new QLabel(" seconds");
    QLabel *lbl02 = new QLabel(" for ");
    QLabel *lbl03 = new QLabel(" seconds");

    QGroupBox *groupBox = new QGroupBox("Alarm actions");

    chkSound = new AlarmCheckBox("Play Sound", key() + "/chkSound", MW->settings, false);
    connect(chkSound, SIGNAL(clicked(bool)), this, SLOT(chkSoundClicked(bool)));

    soundSetter = new FileSetter(mainWindow, "File", "Sounds(*.wav)");
    soundSetter->defaultPath = "C:/Windows/Media";
    soundSetterKey = key() + "/soundSetter";
    soundSetter->setPath(MW->settings->value(soundSetterKey,"C:/Windows/Media/Alarm01.wav").toString());
    connect(soundSetter, SIGNAL(fileSet(const QString&)), this, SLOT(setSoundPath(const QString&)));

    QGroupBox *playBox = new QGroupBox("Play Alarm");
    playOnce = new QRadioButton("Once", playBox);
    playOnceKey = key() + "/playOnce";
    playContinuous = new QRadioButton("Continuous", playBox);
    playOnce->setChecked(MW->settings->value(playOnceKey, false).toBool());
    playContinuous->setChecked(!playOnce->isChecked());
    connect(playOnce, SIGNAL(toggled(bool)), this, SLOT(playOnceToggled(bool)));

    volumeSlider = new QSlider(Qt::Horizontal, this);
    volumeSlider->setRange(0, 128);
    volumeKey = key() + "/volume";
    volumeSlider->setValue(MW->settings->value(volumeKey, 100).toInt());
    connect(volumeSlider, SIGNAL(valueChanged(int)), this, SLOT(volumeChanged(int)));
    QLabel *lbl04 = new QLabel("Volume");

    QGridLayout *pLayout = new QGridLayout();
    pLayout->addWidget(playOnce,        0, 0, 1, 1);
    pLayout->addWidget(playContinuous,  0, 1, 1, 1);
    pLayout->setContentsMargins(11, 6, 11, 6);
    playBox->setLayout(pLayout);

    chkColor = new AlarmCheckBox("Change Color", key() + "/chkColor", MW->settings, false);
    connect(chkColor, SIGNAL(clicked(bool)), this, SLOT(chkColorClicked(bool)));

    btnTest = new QPushButton("Test");
    btnTest->setMaximumWidth(btnTest->fontMetrics().boundingRect(btnTest->text()).width() * 1.5);
    connect(btnTest, SIGNAL(clicked()), this, SLOT(test()));

    chkWrite = new AlarmCheckBox("Write File", key() + "/chkWrite", MW->settings, false);

    dirSetter = new DirectorySetter(mainWindow, "Dir");
    dirSetterKey = key() + "/dirSetter";
    dirSetter->setPath(MW->settings->value(dirSetterKey, QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)).toString());
    connect(dirSetter, SIGNAL(directorySet(const QString&)), this, SLOT(setDirPath(const QString&)));

    QGridLayout *gLayout = new QGridLayout();
    gLayout->addWidget(chkSound,     0, 0, 1, 1);
    gLayout->addWidget(soundSetter,  0, 1, 1, 2);
    gLayout->addWidget(btnTest,      1, 0, 1, 1, Qt::AlignCenter);
    gLayout->addWidget(playBox,      1, 1, 1, 2);
    gLayout->addWidget(lbl04,        2, 1, 1, 1);
    gLayout->addWidget(volumeSlider, 2, 2, 1, 1);
    gLayout->addWidget(chkColor,     3, 0, 1, 1);
    gLayout->addWidget(chkWrite,     4, 0, 1, 1);
    gLayout->addWidget(dirSetter,    4, 1, 1, 2);
    groupBox->setLayout(gLayout);

    QPushButton *close = new QPushButton("Close");

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(chkMin,        0, 0, 1, 1);
    layout->addWidget(minLimit,      0, 1, 1, 1);
    layout->addWidget(lbl00,         0, 2, 1, 1, Qt::AlignRight);
    layout->addWidget(minLimitTime,  0, 3, 1, 1);
    layout->addWidget(lbl01,         0, 4, 1, 1);
    layout->addWidget(chkMax,        1, 0, 1, 1);
    layout->addWidget(maxLimit,      1, 1, 1, 1);
    layout->addWidget(lbl02,         1, 2, 1, 1, Qt::AlignRight);
    layout->addWidget(maxLimitTime,  1, 3, 1, 1);
    layout->addWidget(lbl03,         1, 4, 1, 1);
    layout->addWidget(groupBox,      2, 0, 1, 5);
    layout->addWidget(close,         3, 4, 1, 1);
    setLayout(layout);

    player = new AlarmPlayer(mainWindow);
    player->filename = soundSetter->filename;
    player->audio.volume = volumeSlider->value();
    connect(player, SIGNAL(done()), this, SLOT(soundPlayFinished()));
    connect(MW->control(), SIGNAL(quitting()), this, SLOT(alarmOff()));
    connect(MW->control(), SIGNAL(muting(bool)), this, SLOT(mute(bool)));

    alarmProfile.bl = "#B33525";
    alarmProfile.bm = "#8F0000";
    alarmProfile.bd = "#630000";
    alarmProfile.fl = "#D8DAB5";
    alarmProfile.fm = "#864130";
    alarmProfile.fd = "#735972";
    alarmProfile.sl = "#FFFFFF";
    alarmProfile.sm = "#F4F4E1";
    alarmProfile.sd = "#306294";
}

AlarmPanel::~AlarmPanel()
{
    player->stop();
}

void AlarmPanel::mute(bool muted)
{
    player->mute(muted);
    if (muted)
        volumeSlider->setEnabled(false);
    else
        volumeSlider->setEnabled(true);
}

void AlarmPanel::alarmOff()
{
    cout << "AlarmPanel::alarmOff" << endl;
    if (minAlarmOn || maxAlarmOn) {
        QString str;
        QTextStream(&str) << QTime::currentTime().toString("hh:mm:ss") << "Min Alarm for "
                          << MW->count()->names[obj_id] << " turned off Manually";
        MW->msg(str);
        minAlarmOn = false;
        minAlarmStart = reference;
        MW->applyStyle(MW->config()->getProfile());
        player->stop();
    }
}

void AlarmPanel::feed(int count)
{
    if (first_pass) {
        cout << "First Pass" << endl;
        lastTime = high_resolution_clock::now();
        first_pass = false;
        return;
    }

    auto now = high_resolution_clock::now();
    long msec = duration_cast<milliseconds>(now - lastTime).count();

    if (!k_count.initialized)
        k_count.initialize(count, 0, 0.2f, 0.1f);
    else
        k_count.measure(count, msec);

    lastTime = now;

    if (chkMin->isChecked()) {
        if (k_count.xh00 < minLimit->floatValue()) {
            if (duration_cast<milliseconds>(minAlarmStart - reference).count() == 0) {
                minAlarmStart = now;
            }
            else {
                if (duration_cast<milliseconds>(now - minAlarmStart).count() > minLimitTime->floatValue() * 1000) {
                    if (!minAlarmOn) {
                        QString str;
                        QTextStream(&str) << QTime::currentTime().toString("hh:mm:ss") << "Min Alarm Condition Met for "
                                          << MW->count()->names[obj_id] << " count: " << k_count.xh00;
                        MW->msg(str);
                        minAlarmOn = true;
                        if (chkColor->isChecked()) {
                            MW->applyStyle(alarmProfile);
                        }
                        if (chkSound->isChecked()) {
                            if (player->filename.length() > 0) {
                                QThreadPool::globalInstance()->tryStart(player);
                            }
                        }
                    }
                }
            }
        }
        else {

            minAlarmStart = reference;

            if (minAlarmOn) {
                QString str;
                QTextStream(&str) << QTime::currentTime().toString("hh:mm:ss") << "Min Alarm for "
                                  << MW->count()->names[obj_id] << " turned off";
                MW->msg(str);
                minAlarmOn = false;
                //minAlarmStart = reference;
                MW->applyStyle(MW->config()->getProfile());
                player->stop();
            }
        }
    }

    if (chkMax->isChecked()) {
        if (k_count.xh00 > maxLimit->floatValue()) {
            if (duration_cast<milliseconds>(maxAlarmStart - reference).count() == 0) {
                maxAlarmStart = now;
            }
            else {
                if (duration_cast<milliseconds>(now - maxAlarmStart).count() > maxLimitTime->floatValue() * 1000) {
                    if (!maxAlarmOn) {
                        QString str;
                        QTextStream(&str) << QTime::currentTime().toString("hh:mm:ss") << "Max Alarm Condition Met for "
                                          << MW->count()->names[obj_id] << " count: " << k_count.xh00;
                        MW->msg(str);
                        maxAlarmOn = true;
                        if (chkColor->isChecked()) {
                            MW->applyStyle(alarmProfile);
                        }
                        if (chkSound->isChecked()) {
                            if (player->filename.length() > 0) {
                                QThreadPool::globalInstance()->tryStart(player);
                            }
                        }
                    }
                }
            }
        }
        else {

            maxAlarmStart = reference;

            if (maxAlarmOn) {
                QString str;
                QTextStream(&str) << QTime::currentTime().toString("hh:mm:ss") << "Min Alarm for "
                                  << MW->count()->names[obj_id] << " turned off";
                MW->msg(str);
                maxAlarmOn = false;
                //minAlarmStart = reference;
                MW->applyStyle(MW->config()->getProfile());
                player->stop();
            }
        }
    }

}

void AlarmPanel::chkColorClicked(bool checked)
{
    cout << "AlarmPanel::chkColorClicked" << endl;
    if (minAlarmOn || maxAlarmOn) {
        if (checked)
            MW->applyStyle(alarmProfile);
        else
            MW->applyStyle(MW->config()->getProfile());
    }
}

void AlarmPanel::chkSoundClicked(bool checked)
{
    if (!checked) {
        player->stop();
    }
    else {
        if ((chkMin->isChecked() && minAlarmOn) || (chkMax->isChecked() && maxAlarmOn)) {
            if (player->filename.length() > 0) {
                QThreadPool::globalInstance()->tryStart(player);
            }
        }
    }
}

void AlarmPanel::autoSave()
{
    if (changed) {
        cout << "AlarmPanel::autoSave" << endl;
        MW->settings->setValue(volumeKey, volumeSlider->value());
        changed = false;
    }
}

void AlarmPanel::volumeChanged(int volume)
{
    cout << "AlarmPanel::volumeChanged: " << volume << endl;
    player->audio.volume = volume;
    changed = true;
}

void AlarmPanel::soundPlayFinished()
{
    if (testing) {
        testing = false;
    }
    else {
        if (playContinuous->isChecked() && chkSound->isChecked()) {
            if (minAlarmOn || maxAlarmOn) {
                QThreadPool::globalInstance()->tryStart(player);
            }
        }
    }
}

void AlarmPanel::setSoundPath(const QString& path)
{
    cout << "AlarmPanel::setSoundPath: " << path.toStdString() << endl;
    player->filename = path;
    MW->settings->setValue(soundSetterKey, path);
}

void AlarmPanel::setDirPath(const QString& path)
{
    cout << "AlarmPanel::setDirPath" << path.toStdString() << endl;
    MW->settings->setValue(dirSetterKey, path);
}

void AlarmPanel::playOnceToggled(bool checked)
{
    MW->settings->setValue(playOnceKey, checked);
}

QString AlarmPanel::key() const
{
    return QString("AlarmPanel_%1").arg(obj_id);
}

void AlarmPanel::test()
{
    cout << "AlarmPanel::test" << endl;

    player->stop();

    if (player->filename.length() > 0) {
        QThreadPool::globalInstance()->tryStart(player);
        testing = true;
    }
    else {
        QMessageBox::warning(this, "PlayQt", "No sound file has been specified");
    }
}

AlarmCheckBox::AlarmCheckBox(QString text, QString key, QSettings *settings, bool defaultState) : QCheckBox(text)
{
    this->key = key;
    this->settings = settings;
    setChecked(settings->value(key, defaultState).toBool());
    connect(this, SIGNAL(clicked(bool)), this, SLOT(clicked(bool)));
}

void AlarmCheckBox::clicked(bool arg)
{
    settings->setValue(key, arg);
}

AlarmNumberBox::AlarmNumberBox(QString key, QSettings *settings, QString defaultText)
{
    this->key = key;
    this->settings = settings;
    setText(settings->value(key, defaultText).toString());
    connect(this, SIGNAL(editingFinished()), this, SLOT(edited()));
}

void AlarmNumberBox::edited()
{
    settings->setValue(key, text());
}

AlarmPlayer::AlarmPlayer(QMainWindow *parent)
{
    mainWindow = parent;
    setAutoDelete(false);
}

void AlarmPlayer::audioCallback(void *userData, uint8_t *stream, int streamLength)
{
    AudioData *audio = (AudioData*)userData;
    if (audio->length == 0)
        return;

    uint32_t length = (uint32_t)streamLength;
    length = (length > audio->length ? audio->length : length);
    //SDL_memcpy(stream, audio->position, length);
    memset(stream, 0, length);
    SDL_MixAudioFormat(stream, audio->position, AUDIO_S16SYS, length, audio->volume);
    audio->position += length;
    audio->length -= length;
}

void AlarmPlayer::mute(bool muted)
{
    if (muted) {
        lastVolume = audio.volume;
        audio.volume = 0;
    }
    else {
        audio.volume = lastVolume;
    }

}

void AlarmPlayer::stop()
{
    if (running) {
        audio.length = 0;
        while (running)
            QThread::msleep(10);
    }

}

void AlarmPlayer::run()
{
    running = true;
    SDL_AudioSpec wavSpec;
    uint8_t *wavStart;
    uint32_t wavLength;

    if (SDL_LoadWAV(filename.toLatin1().data(), &wavSpec, &wavStart, &wavLength) == NULL) {
        MW->msg("AlarmPlayer::run - Error loading wav file");
        return;
    }

    audio.position = wavStart;
    audio.length = wavLength;

    wavSpec.callback = audioCallback;
    wavSpec.userdata = &audio;

    if (SDL_OpenAudio(&wavSpec, NULL) < 0) {
        MW->msg(QString("AlarmPlayer::run - Could not open audio: %1").arg(SDL_GetError()));
        return;
    }

    SDL_PauseAudio(0);
    while (audio.length > 0)
        SDL_Delay(100);

    SDL_CloseAudio();
    SDL_FreeWAV(wavStart);
    running = false;
    emit done();
}

AlarmDialog::AlarmDialog(QMainWindow *parent, int obj_id) : PanelDialog(parent)
{
    setWindowTitle("Alarm Configuration");
    panel = new AlarmPanel(mainWindow, obj_id);
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(panel);
    setLayout(layout);

    defaultWidth = 400;
    defaultHeight = 300;
    settingsKey = "AlarmPanel/geometry";
}

AlarmPanel *AlarmDialog::getPanel()
{
    return (AlarmPanel*)panel;
}
