DROP DATABASE IF EXISTS `cs6400_fa22_team090`; 

-- Tables 
CREATE TABLE Household (
  email varchar(100) NOT NULL,
  postal_code varchar(5) NOT NULL,
  home_type varchar(20) NOT NULL,
  square_footage int NOT NULL,
  occupants int NOT NULL, 
  bedrooms int NOT NULL,
  PRIMARY KEY (email));

CREATE TABLE Location(
  postal_code varchar(5) NOT NULL,
  city varchar(30) NOT NULL,
  state varchar(20) NOT NULL,
  latitude float NOT NULL,
  longitude float NOT NULL,
  PRIMARY KEY (postal_code)
);


CREATE TABLE PhoneNumber(
  area_code varchar(3) NOT NULL,
  number varchar(7) NOT NULL,
  phone_type varchar(20) NOT NULL,
  email varchar(100) NOT NULL,
  PRIMARY KEY (area_code, number),
  FOREIGN KEY (email) REFERENCES Household (email));

CREATE TABLE Bathroom (
  email varchar(100) NOT NULL,
  bathroom_order int NOT NULL,
  commodes int NOT NULL,
  bidets int NOT NULL,
  sinks int NOT NULL,
  PRIMARY KEY (email, bathroom_order));

CREATE TABLE HalfBathroom (
  email varchar(100) NOT NULL,
  bathroom_order int NOT NULL,
  nonunique_name varchar(20) NULL,  
  PRIMARY KEY (email,bathroom_order));

CREATE TABLE FullBathroom(
  email varchar(100) NOT NULL,
  bathroom_order int NOT NULL,
  primary_bath BOOLEAN NOT NULL,
  bathtub int NOT NULL,
  shower int NOT NULL,
  tubshower int NOT NULL,
  PRIMARY KEY (email, bathroom_order));

CREATE TABLE Manufacturer(
  manufacturer_name varchar(100) NOT NULL,
  PRIMARY KEY (manufacturer_name));

CREATE TABLE Appliance(
  email varchar(100) NOT NULL,
  appliance_order int NOT NULL,
  manufacturer_name varchar(100) NOT NULL,
  model_name varchar(100) NULL,
  PRIMARY KEY (email, appliance_order),
  FOREIGN KEY (manufacturer_name) REFERENCES Manufacturer (manufacturer_name));

CREATE TABLE TV(
  email varchar(100) NOT NULL,
  appliance_order int NOT NULL,
  display_type varchar(100) NOT NULL,
  display_size float(5) NOT NULL,
  max_resolution  varchar(100) NOT NULL,
  PRIMARY KEY (email,appliance_order));

CREATE TABLE Dryer(
  email varchar(100) NOT NULL,
  appliance_order int NOT NULL,
  heat_source varchar(100) NOT NULL,
  PRIMARY KEY (email,appliance_order));

CREATE TABLE Washer(
  email varchar(100) NOT NULL,
  appliance_order int NOT NULL,
  loading_type varchar(100) NOT NULL,
  PRIMARY KEY (email,appliance_order));

CREATE TABLE RefrigeratorFreezer(
  email varchar(100) NOT NULL,
  appliance_order int NOT NULL,
  refrigerator_type varchar(100) NOT NULL,
  PRIMARY KEY (email,appliance_order));

CREATE TABLE Cooker(
  email varchar(100) NOT NULL,
  appliance_order int NOT NULL,
  PRIMARY KEY (email,appliance_order));

CREATE TABLE Oven(
  email varchar(100) NOT NULL,
  appliance_order int NOT NULL,
  heat_source varchar(100) NOT NULL,
  oven_type varchar(100) NOT NULL,
  PRIMARY KEY (email,appliance_order));

CREATE TABLE Cooktop(
  email varchar(100) NOT NULL,
  appliance_order int NOT NULL,
  heat_source varchar(100) NOT NULL,
  PRIMARY KEY (email,appliance_order));



-- Constraints   Foreign Keys: FK_ChildTable_childColumn_ParentTable_parentColumn


ALTER TABLE Household
  ADD CONSTRAINT fk_Household_postal_code_Location_postal_code FOREIGN KEY (postal_code) REFERENCES Location (postal_code);

ALTER TABLE Bathroom
  ADD CONSTRAINT fk_Bathroom_email_Household_email FOREIGN KEY (email) REFERENCES Household (email);

ALTER TABLE HalfBathroom
  ADD CONSTRAINT fk_HalfBathroom_email_Bathroom_email FOREIGN KEY (email, bathroom_order) REFERENCES Bathroom (email, bathroom_order);

ALTER TABLE FullBathroom
  ADD CONSTRAINT fk_FullBathroom_email_Bathroom_email FOREIGN KEY (email, Bathroom_order) REFERENCES Bathroom (email, bathroom_order); 

ALTER TABLE Appliance
  ADD CONSTRAINT fk_Appliance_email_Household_email FOREIGN KEY (email) REFERENCES Household (email);

ALTER TABLE TV
  ADD CONSTRAINT fk_TV_email_Appliance_email FOREIGN KEY (email, appliance_order) REFERENCES Appliance (email, appliance_order);

ALTER TABLE Dryer
  ADD CONSTRAINT fk_Dryer_email_Appliance_email FOREIGN KEY (email, appliance_order) REFERENCES Appliance (email, appliance_order);

ALTER TABLE Washer
  ADD CONSTRAINT fk_Washer_email_Appliance_email FOREIGN KEY (email, appliance_order) REFERENCES Appliance (email, appliance_order);

ALTER TABLE RefrigeratorFreezer
  ADD CONSTRAINT fk_Refrigerator_email_Appliance_email FOREIGN KEY (email, appliance_order) REFERENCES Appliance (email, appliance_order);

ALTER TABLE Cooker
  ADD CONSTRAINT fk_Cooker_email_Appliance_email FOREIGN KEY (email, appliance_order) REFERENCES Appliance (email, appliance_order);

ALTER TABLE Oven
  ADD CONSTRAINT fk_Oven_email_Cooker_email FOREIGN KEY (email, appliance_order) REFERENCES Appliance (email, appliance_order);

ALTER TABLE Cooktop
  ADD CONSTRAINT fk_Cooktop_email_Cooker_email FOREIGN KEY (email, appliance_order) REFERENCES Appliance (email, appliance_order);







