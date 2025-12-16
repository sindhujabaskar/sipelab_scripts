#include <CapacitiveSensor.h>

// Solenoid / Switch (Button) Pins
const int switchPin = 8; // manual trigger
const int solenoidPin = 9; // transistor input to drive solenoid
int switchState = 0;

// Capacitive Circuit
const int sendPin = 13;
const int receivePin = 12;
CapacitiveSensor Spout = CapacitiveSensor(sendPin,receivePin);

// capacitive threshold -- #needto calibrate
long capacitiveThreshold = 20;

// pulse duration in ms (controls how much water)
unsigned long pulseLength = 35;   // ← make smaller for less water


// track input state to detect NEW triggers
bool previouslyTriggered = false;
void setup() {
  pinMode(solenoidPin, OUTPUT);
  pinMode(switchPin, INPUT);
  digitalWrite(solenoidPin, LOW);
  Serial.begin(115200);
}
void loop() {
  long total = Spout.capacitiveSensor(10);
  bool capacitanceTriggered = (total >= capacitiveThreshold);
  switchState = digitalRead(switchPin);
  bool switchTriggered = (switchState == HIGH);
  // TRUE if either sensor OR switch is triggered
  bool triggered = (switchTriggered || capacitanceTriggered);
  // --- PULSE LOGIC ADDED HERE ---
  if (triggered && !previouslyTriggered) {
      // NEW trigger → pulse solenoid
      digitalWrite(solenoidPin, HIGH);
      delay(pulseLength);      // solenoid open briefly
      digitalWrite(solenoidPin, LOW);
  }
  // --------------------------------
  previouslyTriggered = triggered;
  // Serial logging
  Serial.print(total);
  Serial.print(“\t”);
  Serial.println(triggered ? 1 : 0);
  delay(20);
}