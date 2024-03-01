-- Top 25 Manufacturers repot --

SELECT manufacturer_name, COUNT(manufacturer_name) AS "Count" FROM Appliance
GROUP BY manufacturer_name 
ORDER BY COUNT(manufacturer_name) DESC LIMIT 25 ;

-- Drill down report based on clicked manufacturer --
SELECT manufacturer_name FROM Manufacturer;
-- WHERE manufacturer_name = $manufacturer; -- will need to add this line when setting up to front end


SELECT Type, COUNT(Type) AS "COUNT" FROM
(
SELECT email, appliance_order, 'Cooker' AS Type FROM Cooker
 UNION
SELECT email, appliance_order, 'TV' AS Type FROM TV
UNION
SELECT email, appliance_order, 'Washer' AS Type FROM Washer
UNION
SELECT email, appliance_order, 'Dryer' AS Type FROM Dryer
UNION
SELECT email, appliance_order, 'refrigerator/Freezer' AS Type FROM
RefrigeratorFreezer
) as A
INNER JOIN Appliance 
ON A.email = Appliance.email AND A.appliance_order = Appliance.appliance_order
WHERE Appliance.manufacturer_name = 'Adkililar Holdings Company' -- will need to change this line to accept user selection
GROUP BY Type;

-- Manufacturer model search --

SELECT distinct CONCAT(UPPER(SUBSTRING(manufacturer_name,1,1)),
LOWER(SUBSTRING(manufacturer_name, 2))) AS manufacturer_name,
CONCAT(UPPER(SUBSTRING(model_name, 1,1)), LOWER(SUBSTRING(model_name,2)))
AS model_name
FROM (SELECT LOWER(manufacturer_name) AS manufacturer_name,
COALESCE(LOWER(model_name), '') AS model_name FROM Appliance
WHERE lower(manufacturer_name) LIKE LOWER('%dime%')  -- will need to change this line to accept user selection
OR lower(model_name) LIKE LOWER('%dime%')) A  -- will need to change this line to accept user selection
ORDER BY manufacturer_name, model_name;

-- Avg TV Size by State --
SELECT state, Round(cast(AVG(display_size) as numeric),1) FROM Household
INNER JOIN Location
ON Household.postal_code = Location.postal_code
INNER JOIN TV
ON Household.email = TV.email
GROUP BY state
ORDER BY state;

-- Drill Down Report for Avg TV Size -- 
select distinct state from location
where state = 'FL'; -- will need to change this line to accept user selection

SELECT display_type, max_resolution, ROUND(CAST(AVG(display_size) AS NUMERIC),1) FROM
Household
INNER JOIN Location
ON Household.postal_code = Location.postal_code
INNER JOIN TV
ON Household.email = TV.email
WHERE state = 'FL'
GROUP BY display_type, max_resolution
ORDER BY AVG(display_size) DESC;

---------- Extra Fridge/Freezer report --------
SELECT COUNT(email) FROM(
SELECT email, COUNT(*) FROM RefrigeratorFreezer
GROUP BY email
HAVING COUNT(*) > 1
) A;

-- Get list of emails with more than 1 fridge --
WITH email_list AS(
SELECT email FROM(
SELECT email, COUNT(*) AS fridge_count FROM RefrigeratorFreezer
GROUP BY email
HAVING COUNT(*) > 1 

)a )
,

total_list AS (
select state, count(*) as household_count from
(select email, state, count(*) from refrigeratorfreezer natural join household
natural join location
group by email, state
having count(*) > 1) b
group by state
order by count(*) desc
)
,

chest_list as (
select state, count(distinct email) as chest_count from(
select email, state, refrigerator_type from refrigeratorfreezer natural join household
natural join location
where refrigerator_type = 'chest freezer'
and email in (select email from email_list))c
group by state
)
,
upright_list as(
select state, count(distinct email) as upright_count from(
select email, state, refrigerator_type from refrigeratorfreezer natural join household
natural join location
where refrigerator_type =  'upright freezer'
and email in (select email from email_list))d
group by state
)
,

other_list as(
select state, count(distinct email) as other_count from(
select email, state, refrigerator_type from refrigeratorfreezer natural join household
natural join location
where refrigerator_type <> 'upright freezer' and refrigerator_type <> 'chest freezer'
and email in (select email from email_list))e
group by state
)


select state, household_count,
ROUND(cast(chest_count as numeric) / cast(household_count as numeric) * 100) AS chest_freezer_perc,
ROUND(cast(upright_count as numeric) / cast(household_count as numeric) * 100) AS upright_freezer_perc,
ROUND(cast(other_count as numeric) / cast(household_count as numeric) * 100) AS other_freezer_perc

from total_list
natural join chest_list
natural join upright_list
natural join other_list

order by household_count desc
limit 10;
---------- Extra Fridge/Freezer report --------

---------- Laundry Center report --------
-- Combine household, washer, dryer and location tables together to get washer’s and dryer’s by household by state --
WITH wash_dry AS(
SELECT Household.*,Location.state,Washer.loading_type,Dryer.heat_source FROM
Household
INNER JOIN Location
ON Household.postal_code = Location.postal_code
LEFT JOIN Washer
ON Household.email = Washer.email
LEFT JOIN Dryer
ON Household.email = Dryer.email)
,
-- Get most popular washer by state by getting counts by state and loading type, then adding a new column with maximum count by state. --
wash_count AS(
SELECT * FROM(
SELECT *, MAX(wash_count) OVER(PARTITION BY state ORDER BY state) AS
max_wash_count
FROM(
SELECT state,loading_type, COUNT(*) AS wash_count FROM wash_dry
GROUP BY state, loading_Type
) w
) x
WHERE wash_count = max_wash_count
)
,
-- Get most popular dryer by state by getting counts by state and dryer heat source,then adding a new column with maximum count by state. --
dry_count AS (
SELECT * FROM(
SELECT *, MAX(dry_count) OVER(PARTITION BY state ORDER BY state) AS
max_dry_count
FROM(
SELECT state,heat_source, COUNT(*) AS dry_count FROM wash_dry
GROUP BY state, heat_source
) d
) x
WHERE dry_count = max_dry_count
)
-- combine wash_count table and dry_count table --
select state,loading_type,heat_source from (
SELECT wash_count.state, wash_count.loading_Type,dry_count.heat_source, row_number() over(partition by wash_count.state) as row_num FROM
wash_count
INNER JOIN dry_count
ON wash_count.state = dry_count.state
	) a
where row_num = 1;
---------- Laundry Center report --------

---------- Laundry Center report 2 --------
-- Get household counts by state by joining household, location, washer, and dryer tables together then grouping by state and counting all households with a washer but no dryer. --
select state, count(*) from household natural join location
where email in (select email from washer except select email from dryer)
group by state
order by count(*) desc, state;

---------- Laundry Center report 2 --------

--------- Bathroom Stats __________________
SELECT MIN(bath_counts), ROUND(AVG(bath_counts),1), MAX(bath_counts) FROM
(SELECT COUNT(*) AS bath_counts FROM Bathroom GROUP BY email) a;
SELECT MIN(half_bathcounts), ROUND(AVG(half_bathcounts),1), MAX(half_bathcounts)
FROM
(SELECT COUNT(*) AS half_bathcounts FROM HalfBathroom GROUP BY email) a;
SELECT MIN(full_bathcounts), ROUND(AVG(full_bathcounts),1), MAX(full_bathcounts)
FROM
(SELECT COUNT(*) AS full_bathcounts FROM FullBathroom GROUP BY email) a;
SELECT MIN(commodes_counts), ROUND(AVG(commodes_counts),1),
MAX(commodes_counts) FROM
(SELECT SUM(commodes) AS commodes_counts FROM Bathroom GROUP BY email) a;
SELECT MIN(sinks_counts), ROUND(AVG(sinks_counts),1), MAX(sinks_counts) FROM
(SELECT SUM(sinks) AS sinks_counts FROM Bathroom GROUP BY email) a;
SELECT MIN(bidets_counts), ROUND(AVG(bidets_counts),1), MAX(bidets_counts) FROM
(SELECT SUM(bidets) AS bidets_counts FROM Bathroom GROUP BY email) a;
SELECT MIN(bathtubs_counts), ROUND(AVG(bathtubs_counts),1), MAX(bathtubs_counts)
FROM
(SELECT SUM(bathtub) AS bathtubs_counts FROM FullBathroom GROUP BY email) a;
SELECT MIN(shower_counts), ROUND(AVG(shower_counts),1), MAX(shower_counts)
FROM
(SELECT SUM(shower) AS shower_counts FROM FullBathroom GROUP BY email) a;
SELECT MIN(tubshower_counts), ROUND(AVG(tubshower_counts),1),
MAX(tubshower_counts) FROM
(SELECT SUM(tubshower) AS tubshower_counts FROM FullBathroom GROUP BY email) a;
SELECT state, SUM(bidets) as bidets_count FROM Bathroom INNER JOIN Household ON
Bathroom.email = Household.email
INNER JOIN Location ON Household.postal_code = Location.postal_code 
GROUP BY state
ORDER BY SUM(bidets) DESC LIMIT 1;

SELECT postal_code, SUM(bidets) as bidets_count FROM Bathroom INNER JOIN
Household ON Bathroom.email = Household.email
GROUP BY postal_code
ORDER BY SUM(bidets) DESC LIMIT 1;

select count(*) from (
SELECT COUNT(*) FROM Bathroom
WHERE email in (SELECT email FROM FullBathroom WHERE primary_bath = True)
GROUP BY email
HAVING COUNT(*) = 1) a;

------- POSTAL Code ------------------

with postal as(
select '78727' as postal_code
),

users as (
select postal_code,longitude,latitude, 50 as radius 
	from location
	where postal_code in (select postal_code from postal)
)


SELECT ROUND(cast(AVG(Bath_count) as numeric),1) FROM
(SELECT COUNT(*) as Bath_count FROM
(SELECT email, 2 * 3958.75 * atan2(sqrt(abs(a)), sqrt(1-a)) as d FROM
 (SELECT email, (sin(0.5 * (location.latitude - (select users.latitude from users)) *PI()/180) * sin(0.5 * (location.latitude -
(select users.latitude from users)) * PI()/180) + cos(location.latitude*PI()/180) * cos((select users.latitude from users)*PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) *
PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * PI()/180)) as a
FROM Bathroom 
Natural Join Household
Natural Join Location
 left JOIN users
  ON Household.postal_code = users.postal_code
 ) c
) e
WHERE d <= (select radius from users)
GROUP BY email) f;

---------------------------------------------------------------------

with postal as(
select '78727' as postal_code
),

users as (
select postal_code,longitude,latitude, 100 as radius 
	from location
	where postal_code in (select postal_code from postal)
)

SELECT ROUND(cast(AVG(bedrooms) as numeric), 1) as bedroom, round(cast(AVG(occupants) as numeric), 0) as occupants FROM
(SELECT email, bedrooms, occupants, 2 * 3958.75 * atan2(sqrt(abs(a)), sqrt(1-a)) as d FROM
 (SELECT email, bedrooms, occupants, (sin(0.5 * (location.latitude - (select users.latitude from users)) *PI()/180) * sin(0.5 * (location.latitude -
(select users.latitude from users)) * PI()/180) + cos(location.latitude * PI()/180) * cos((select users.latitude from users)* PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) *
PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * PI()/180)) as a
FROM household INNER JOIN location
ON Household.postal_code = location.postal_code 
 left JOIN users
  ON Household.postal_code = users.postal_code
 ) c
) e
WHERE d <= (select radius from users);
---------------------------------------------------------------------
with postal as(
select '78727' as postal_code
),

users as (
select postal_code,longitude,latitude, 300 as radius 
	from location
	where postal_code in (select postal_code from postal)
)

select ROUND(SUM(occupants)/SUM(commodes_count),2) as ratio from
(select email, occupants, sum(commodes) as commodes_count from
(select email, occupants, commodes,(2 * 3958.75 * atan2(sqrt(abs(a)), sqrt(1-a))) as d from
(SELECT email, occupants, commodes, (sin(0.5 * (location.latitude - (select users.latitude from users)) *PI()/180) * sin(0.5 * (location.latitude -
(select users.latitude from users)) * PI()/180) + cos(location.latitude * PI()/180) * cos((select users.latitude from users)* PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) *
PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * PI()/180)) as a
FROM bathroom Natural Join household
 Natural Join location
 left JOIN users
  ON household.postal_code = users.postal_code) c) e
  where d <= (select radius from users)
 group by email, occupants) f;


---------------------------------------------------------------------
with postal as(
select '14043' as postal_code
),

users as (
select postal_code,longitude,latitude, 200 as radius 
	from location
	where postal_code in (select postal_code from postal)
)

 SELECT ROUND(cast(AVG(appliance_count) as numeric),1) FROM
(SELECT COUNT(*) as appliance_count FROM
(SELECT email, 2 * 3958.75 * atan2(sqrt(abs(a)), sqrt(1-a)) as d FROM
 (SELECT email, (sin(0.5 * (location.latitude - (select users.latitude from users)) *PI()/180) * sin(0.5 * (location.latitude -
(select users.latitude from users)) * PI()/180) + cos(location.latitude*PI()/180) * cos((select users.latitude from users)*PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) *
PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * PI()/180)) as a
FROM Appliance 
Natural Join Household
Natural Join Location
 left JOIN users
  ON Household.postal_code = users.postal_code
 ) c
) e
WHERE d <= (select radius from users)
GROUP BY email) f;
---------------------------------------------------------------------
with postal as(
select '14043' as postal_code
),

users as (
select postal_code,longitude,latitude, 100 as radius 
	from location
	where postal_code in (select postal_code from postal)
)


SELECT heat_source FROM
((SELECT email, heat_source FROM
Dryer)
UNION
(SELECT email, heat_source FROM
Oven) 
UNION
(SELECT email, heat_source FROM
Cooktop)) a1
where email IN
(SELECT email FROM
(SELECT email, 2 * 3958.75 * atan2(sqrt(abs(a)), sqrt(1-a)) as d FROM
 (SELECT email, (sin(0.5 * (location.latitude - (select users.latitude from users)) *PI()/180) * sin(0.5 * (location.latitude -
(select users.latitude from users)) * PI()/180) + cos(location.latitude*PI()/180) * cos((select users.latitude from users)*PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) *
PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * PI()/180)) as a
FROM Appliance 
Natural Join Household
Natural Join Location
 left JOIN users
  ON Household.postal_code = users.postal_code
 ) c
) e
where d <= (select radius from users))
GROUP By heat_source
ORDER BY COUNT(*) DESC
LIMIT 1;

 

