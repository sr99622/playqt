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
    FileSetter(QMainWindow *parent, const QString& labelText, const QString& filter);
    void setPath(const QString& path);
    void setPath();

    QLabel *label;
    QLineEdit *text;
    QPushButton *button;
    QString filename;
    QString filter;
    QString defaultPath;

    QMainWindow *mainWindow;

signals:
    void fileSet(const QString&);

public slots:
    void selectFile();

};

#endif // FILESETTER_H
