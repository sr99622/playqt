#ifndef ALARMPANEL_H
#define ALARMPANEL_H

#include <QCheckBox>
#include <QRunnable>
#include <QRadioButton>
#include <QMutex>
#include <QSettings>

#include "Utilities/kalman.h"
#include "Utilities/paneldialog.h"
#include "Utilities/numbertextbox.h"
#include "Utilities/filesetter.h"
#include "Utilities/directorysetter.h"

#include <chrono>

using namespace std::chrono;

struct AudioData
{
    uint8_t *position;
    uint32_t length;
};

class AlarmPlayer : public QObject, public QRunnable
{
    Q_OBJECT

public:
    AlarmPlayer(QMainWindow *parent);
    void run() override;
    void setLength(int length);
    void stop();
    static void audioCallback(void *userData, uint8_t *stream, int streamLength);

    QMainWindow *mainWindow;
    QString filename;
    AudioData audio;
    QMutex mutex;
    bool running = false;

signals:
    void done();

};

class AlarmCheckBox : public QCheckBox
{
    Q_OBJECT

public:
    AlarmCheckBox(QString text, QString key, QSettings *settings, bool defaultState);

    QString key;
    QSettings *settings;

public slots:
    void clicked(bool);

};

class AlarmNumberBox : public NumberTextBox
{
    Q_OBJECT

public:
    AlarmNumberBox(QString key, QSettings *settings, QString defaultText);

    QString key;
    QSettings *settings;

public slots:
    void edited();

};

class AlarmPanel : public Panel
{
    Q_OBJECT

public:
    AlarmPanel(QMainWindow *parent, int obj_id);
    ~AlarmPanel();
    void feed(int count);
    QString key() const;

    int obj_id;
    AlarmNumberBox *minLimit;
    AlarmNumberBox *maxLimit;
    AlarmNumberBox *minLimitTime;
    AlarmNumberBox *maxLimitTime;
    AlarmCheckBox *chkMin;
    AlarmCheckBox *chkMax;
    AlarmCheckBox *chkSound;
    AlarmCheckBox *chkWrite;
    AlarmCheckBox *chkColor;
    FileSetter *soundSetter;
    DirectorySetter *dirSetter;
    QRadioButton *playOnce;
    QRadioButton *playContinuous;
    QPushButton *btnTest;

    bool minAlarmOn = false;
    bool maxAlarmOn = false;
    high_resolution_clock::time_point minAlarmStart;
    high_resolution_clock::time_point maxAlarmStart;
    high_resolution_clock::time_point reference;

    Kalman k_count;
    bool first_pass = true;
    high_resolution_clock::time_point t1;

    AlarmPlayer *player;

    QString soundSetterKey;
    QString dirSetterKey;
    QString playOnceKey;

    bool testing = false;

public slots:
    void test();
    void setSoundPath(const QString&);
    void setDirPath(const QString&);
    void playOnceToggled(bool);
    void soundPlayFinished();
    void chkSoundClicked(bool);
    void chkColorClicked(bool);

};

class AlarmDialog : public PanelDialog
{
    Q_OBJECT

public:
    AlarmDialog(QMainWindow *parent, int obj_id);
    AlarmPanel *getPanel();

};

#endif // ALARMPANEL_H
