#ifndef COLORBUTTON_H
#define COLORBUTTON_H

#include <QMainWindow>
#include <QColor>
#include <QCheckBox>
#include <QPushButton>

class ColorButton : public QWidget
{
    Q_OBJECT

public:
    ColorButton(const QString& qss_name, const QString& color_name);
    QString getStyle() const;

    QString name;
    QColor color;
    QPushButton *button;
    QString settingsKey;

public slots:
    void clicked();


};

#endif // COLORBUTTON_H
