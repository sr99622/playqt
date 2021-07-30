#include "alarmpanel.h"
#include "mainwindow.h"

AlarmPanel::AlarmPanel(QMainWindow *parent, int obj_id) : Panel(parent)
{
    this->obj_id = obj_id;

    minLimit = new NumberTextBox();
    maxLimit = new NumberTextBox();
    minLimitTime = new NumberTextBox();
    maxLimitTime = new NumberTextBox();
    int boxWidth = minLimit->fontMetrics().boundingRect("000000").width();
    minLimit->setMaximumWidth(boxWidth);
    maxLimit->setMaximumWidth(boxWidth);
    minLimitTime->setMaximumWidth(boxWidth);
    maxLimitTime->setMaximumWidth(boxWidth);
    chkMin = new QCheckBox("Alarm if count goes below: ");
    chkMax = new QCheckBox("Alarm if count goes above: ");
    QLabel *lbl00 = new QLabel(" for ");
    QLabel *lbl01 = new QLabel(" seconds");
    QLabel *lbl02 = new QLabel(" for ");
    QLabel *lbl03 = new QLabel(" seconds");

    QGroupBox *groupBox = new QGroupBox("Alarm actions");
    chkBeep = new QCheckBox("System Beep");
    chkWrite = new QCheckBox("Write File");
    chkColor = new QCheckBox("Change Color");
    QGridLayout *gLayout = new QGridLayout();
    gLayout->addWidget(chkBeep,  0, 0, 1, 1);
    gLayout->addWidget(chkColor, 0, 1, 1, 1);
    gLayout->addWidget(chkWrite, 0, 2, 1, 1);
    groupBox->setLayout(gLayout);

    QPushButton *close = new QPushButton("Close");

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(chkMin,        0, 0, 1, 1);
    layout->addWidget(minLimit,      0, 1, 1, 1);
    layout->addWidget(lbl00,         0, 2, 1, 1);
    layout->addWidget(minLimitTime,  0, 3, 1, 1);
    layout->addWidget(lbl01,         0, 4, 1, 1);
    layout->addWidget(chkMax,        1, 0, 1, 1);
    layout->addWidget(maxLimit,      1, 1, 1, 1);
    layout->addWidget(lbl02,         1, 2, 1, 1);
    layout->addWidget(maxLimitTime,  1, 3, 1, 1);
    layout->addWidget(lbl03,         1, 4, 1, 1);
    layout->addWidget(groupBox,      2, 0, 1, 5);
    layout->addWidget(close,         3, 4, 1, 1);
    setLayout(layout);
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
