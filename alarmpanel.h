#ifndef ALARMPANEL_H
#define ALARMPANEL_H

#include <QCheckBox>

#include "Utilities/paneldialog.h"
#include "Utilities/numbertextbox.h"

class AlarmPanel : public Panel
{
    Q_OBJECT

public:
    AlarmPanel(QMainWindow *parent, int obj_id);

    int obj_id;
    NumberTextBox *minLimit;
    NumberTextBox *maxLimit;
    NumberTextBox *minLimitTime;
    NumberTextBox *maxLimitTime;
    QCheckBox *chkMin;
    QCheckBox *chkMax;
    QCheckBox *chkBeep;
    QCheckBox *chkWrite;
    QCheckBox *chkColor;

};

class AlarmDialog : public PanelDialog
{
    Q_OBJECT

public:
    AlarmDialog(QMainWindow *parent, int obj_id);

};

#endif // ALARMPANEL_H
