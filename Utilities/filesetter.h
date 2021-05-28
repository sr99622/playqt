#ifndef FILESETTER_H
#define FILESETTER_H

#include <QWidget>
#include <QMainWindow>
#include <QObject>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>

class FileSetter : public QWidget
{
    Q_OBJECT

public:
    FileSetter(QMainWindow *parent, QString labelText, QString filter);
    void setPath(QString path);
    void trimHeight();

    QLabel *label;
    QLineEdit *text;
    QPushButton *button;
    QString filename;
    QString filter;

    QMainWindow *mainWindow;

signals:
    void fileSet(QString path);

public slots:
    void selectFile();

};

#endif // FILESETTER_H
