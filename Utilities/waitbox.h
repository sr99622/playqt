#ifndef WAITBOX_H
#define WAITBOX_H

#include <QMainWindow>
#include <QDialog>

class WaitBox : public QDialog
{
    Q_OBJECT

public:
    WaitBox(QMainWindow *parent);

};

#endif // WAITBOX_H
