from flask import Flask, render_template, request, redirect, url_for, jsonify
import psycopg2
import psycopg2.extras
import json
import jinja2
from markupsafe import Markup

env = jinja2.Environment()
env.globals.update(zip=zip)

 
app = Flask(__name__)
app.secret_key = "*********"
app.jinja_env.filters['zip'] = zip
postalCode_from_db = None
postCode_error_msg = None
turn_off_isPrimary = False

DB_HOST = "localhost"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "*******"

temp_user = {}
bathroom_order = 0
household_bathrooms = []
user_postalcode = None
searchRadius = 0

household_appliance = []
bathroom_order = 0
appliance_order = 0
 
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)


@app.route('/')
def index():


    return render_template('html/MainMenu.html')

@app.route('/household', methods=['POST', 'GET'])
def household_email():
    global temp_user, household_bathrooms, household_appliance, bathroom_order, appliance_order
    temp_user = {}
    household_bathrooms = []
    household_appliance = []
    bathroom_order = 0
    appliance_order = 0

    if request.method == 'POST':
        email = request.form['email']
        cur.execute('SELECT email FROM household WHERE email=%s',(email,))
        email_from_db = cur.fetchall()


        if len(email_from_db) == 0:
            temp_user["email"]=email

            return redirect(url_for("postal_code"))
        else:
            dup_email_error = 'Email Already exists. Please enter a new one.'
            return render_template('html/get_email_address.html', dup_email_error=dup_email_error)

    return render_template('html/get_email_address.html')


@app.route('/postal_code', methods=['POST','GET'])
def postal_code():

    if request.method == 'POST':
        postalCode = request.form['postalCode']
        cur.execute('SELECT postal_code, city, state FROM Location WHERE postal_code =%s', (postalCode,))
        postalCode_from_db = cur.fetchall()
        if len(postalCode_from_db) != 0:
            temp_user['postal_code'] = postalCode_from_db[0][0]
            temp_user['city'] = postalCode_from_db[0][1]
            temp_user['state'] = postalCode_from_db[0][2]


            return render_template('html/postalcode_verification.html', postalCode_from_db=postalCode_from_db)
        else:
            postCode_error_msg = 'The Zip Code you entered does not exist. Please Enter a new one.'
        return render_template("html/postal_code_entry.html",postCode_error_msg=postCode_error_msg)

    return render_template('html/postal_code_entry.html')


@app.route('/verifypostalcode')
def verify_postal_code():

    return render_template('html/postalcode_verification.html', postalCode_from_db=postalCode_from_db, postCode_error_msg=postCode_error_msg)

@app.route('/phone_number')
def phone_number():
    return render_template('html/phoneNum.html')

@app.route('/phone_number_entry', methods=['POST', 'GET'])
def phone_number_entry():

    if request.method == 'POST':

        areaCode = request.form['areaCode']
        number = request.form['number']

        phoneType = request.form['phoneType']


        cur.execute("SELECT area_code, number from phonenumber where area_code = %s and number = %s", (areaCode,number,))
        phone_number_from_db = cur.fetchall()


        if len(phone_number_from_db) == 0:
            temp_user ["phone"] = {"area_code":areaCode,
                                   "number":number,
                                   "phone_type":phoneType}
        else:
            error_message = "The Phone number already exists. Please enter a new one."
            return render_template('html/phone_number_entry.html', error_message=error_message)


        return redirect(url_for('household_info_entry'))
    return render_template('html/phone_number_entry.html')

@app.route('/household_info_entry', methods=['POST', 'GET'])
def household_info_entry():


    #check to see if the temp user has number in it, and it not then Null is inserted
    if 'phone' not in temp_user:
        temp_user["phone"] = {"area_code": None,
                              "number": None,
                              "phone_type": None}


    if request.method == 'POST':
        homeType = request.form['homeType']
        squareFootage = request.form['squrefootage']
        occupants = request.form['occupants']
        bedrooms = request.form['bedrooms']

        temp_user["home_type"] = homeType
        temp_user["square_footage"]=squareFootage
        temp_user["occupants"]=occupants
        temp_user["bedrooms"] = bedrooms

        return redirect(url_for('add_bathroom'))
    return render_template('html/household_info.html')

@app.route('/add_bathroom', methods=['GET','POST'])
def add_bathroom():
    global bathroom_order, turn_off_isPrimary


    if request.method == 'POST':
        sinks = int(request.form['sinks'])
        commodes = int(request.form['commodes'])
        bidets = int(request.form['bidets'])

        try:
            bathtubs = int(request.form['bathtubs'])
            showers = int(request.form['showers'])
            tubs = int(request.form['tubs'])
        except KeyError:
            bathtubs = None
            showers = None
            tubs = None

        bathroom_order +=1
        try:
            request.form['isPrimary']
        except KeyError:
            isPrimary = False
        else:
            isPrimary = request.form['isPrimary']
            isPrimary = bool(isPrimary)
            turn_off_isPrimary = True

        if bathtubs == None and showers == None and tubs == None:
            bathroom_type = "half"
        else:
            bathroom_type = "full"

        if bathroom_type == "half":
            nonunique_name = request.form['nonunique_name']
        else:
            nonunique_name = "NULL"

        print(f"{sinks}, {commodes}, {bidets}, {bathtubs}, {showers}, {tubs}, {bathroom_type}")
        if bathroom_type == "half":
            if sinks + commodes + bidets < 1:
                bathroom_order-=1
                error_msg = "Select at least ONE(1) sink or commode or bidet for Half Bathroom."
                return render_template('html/add_bathroom.html', error_msg=error_msg, turn_off_isPrimary=turn_off_isPrimary)
        else:
            if (showers + tubs + bathtubs < 1) or (sinks + commodes + bidets < 1):
                bathroom_order -= 1
                error_msg = "Select at least ONE(1) bathtub or shower or tub/shower & at least ONE(1) sink or commode or bidet for Full Bathroom."
                return render_template('html/add_bathroom.html', error_msg=error_msg, turn_off_isPrimary=turn_off_isPrimary)

        new_bathroom = {
            "bathroom_order":bathroom_order,
            "sinks":sinks,
            "commodes":commodes,
            "bidets":bidets,
            "bathtubs":bathtubs,
            "showers":showers,
            "tubs":tubs,
            "bathroom_type":bathroom_type,
            "isPrimary":isPrimary,
            "nonunique_name":nonunique_name
        }
        print(new_bathroom)

        household_bathrooms.append(new_bathroom)

        temp_user["bathrooms"]=household_bathrooms


        return redirect(url_for('bathroom_listing',household_bathrooms=household_bathrooms, turn_off_isPrimary=turn_off_isPrimary ))

    return render_template('html/add_bathroom.html', turn_off_isPrimary=turn_off_isPrimary)

@app.route('/bathroom_listing')
def bathroom_listing():

    return render_template("html/bathroom_listing.html", household_bathrooms=household_bathrooms)

@app.route('/add_appliance', methods=['POST', 'GET'])
def add_appliance():
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute('SELECT * from manufacturer')
    manufs = cur.fetchall()



    global household_appliance, appliance_order
    if request.method == 'POST':
        appliance_order += 1
        appliance_type = request.form['appliance-type']
        model_name = request.form.get('model-name')


        manuf = request.values['manufacturer']


        manuf = request.form['manufacturer']
        if (appliance_type == 'TV'):
            display_type = request.form['display-type']
            display_size = request.form['display-size']
            max_resolution = request.form['max-resolution']
            new_appliance = {'appliance_order':appliance_order, 'model_name':model_name, 'appliance_type':appliance_type, 'manuf':manuf, 'display_type':display_type, \
                'display_size': display_size, 'max_resolution':max_resolution}

        if (appliance_type == 'Refrigerator/freezer'):
            refrigerator_type = request.form['refrigerator_type']
            new_appliance = {'appliance_order':appliance_order, 'appliance_type':appliance_type, 'manuf':manuf, 'model_name':model_name,'refrigerator_type':refrigerator_type}

        if (appliance_type == 'Dryer'):
            dryer_heatsource = request.form['dryer_heatsource']
            new_appliance = {'appliance_order':appliance_order, 'appliance_type':appliance_type, 'manuf':manuf, 'model_name':model_name,'dryer_heatsource':dryer_heatsource}

        if (appliance_type == 'Washer'):
            washer_loadtype = request.form['washer_loadtype']
            new_appliance = {'appliance_order':appliance_order, 'appliance_type':appliance_type, 'manuf':manuf, 'model_name':model_name,'washer_loadtype':washer_loadtype}

        if (appliance_type == 'Cooker'):
            oven_checked = request.form.get('oven')
            cooktop_checked = request.form.get('cooktop')
            cooker_type = []
            oven_heatsource = []
            cooktop_heatsource = 'null'
            oven_type = 'null'
            cooktop_heatsource = 'null'

            if oven_checked:
                cooker_type.append('oven')
                gas_checked = request.form.get('gas')
                electric_checked = request.form.get('electric')
                microwave_checked = request.form.get('microwave')
                oven_type = request.form['oven_type']

                if gas_checked:
                    oven_heatsource.append('gas')
                if electric_checked:
                    oven_heatsource.append('electric')
                if microwave_checked:
                    oven_heatsource.append('microwave')
            if cooktop_checked:
                cooker_type.append('cooktop')
                cooktop_heatsource = request.form.get('cooktop_heatsource')
            new_appliance = {'appliance_order':appliance_order, 'appliance_type':appliance_type, 'manuf':manuf, 'model_name':model_name,'cooker_type': cooker_type, 'oven_heatsource':oven_heatsource, 'oven_type':oven_type, 'cooktop_heatsource': cooktop_heatsource}

        household_appliance.append(new_appliance)
        temp_user['household_appliances']=household_appliance

        temp_user_json = json.dumps(temp_user, indent=4)
        with open("user.json","w") as outfile:
            outfile.write(temp_user_json)

        return render_template ('html/appliance_listing.html', household_appliance=household_appliance)

    return render_template('html/add_appliance.html', manufs=manufs)


@app.route('/list_appliance', methods=['POST', 'GET'])
def list_appliance():
    return render_template('html/appliance_listing.html', household_appliance=household_appliance)


@app.route('/wrap_up', methods=['POST', 'GET'])
def wrap_up():
    mycursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    sql = "INSERT INTO Household (email, postal_code,home_type,square_footage,occupants,bedrooms) VALUES (%s, %s,%s, %s,%s, %s)"
    val = (temp_user['email'],temp_user['postal_code'],temp_user['home_type'],temp_user['square_footage'],temp_user['occupants'],temp_user['bedrooms'],)
    mycursor.execute(sql, val)

    if temp_user['phone']['area_code']:

        sql = "INSERT INTO PhoneNumber (area_code, number, phone_type, email) VALUES (%s, %s,%s, %s)"
        val = (temp_user['phone']['area_code'],temp_user['phone']['number'],temp_user['phone']['phone_type'],temp_user['email'],)
        mycursor.execute(sql, val)


    for i in range(len(temp_user['bathrooms'])):
        current_bathroom = temp_user['bathrooms'][i]
        sql1 = "INSERT INTO Bathroom (email, bathroom_order, commodes, bidets, sinks) VALUES (%s, %s,%s, %s,%s)"
        val1 = (temp_user['email'],current_bathroom['bathroom_order'],current_bathroom['commodes'],current_bathroom['bidets'],current_bathroom['sinks'],)
        mycursor.execute(sql1, val1)
        if current_bathroom['bathroom_type'] == 'half':
            sql2 = "INSERT INTO HalfBathroom (email, bathroom_order, nonunique_name) VALUES (%s, %s,%s)"
            val2 = (temp_user['email'],current_bathroom['bathroom_order'], current_bathroom['nonunique_name'],)
            mycursor.execute(sql2, val2)
        else:
            sql3 = "INSERT INTO FullBathroom (email, bathroom_order, primary_bath, bathtub, shower, tubshower) VALUES (%s, %s,%s, %s,%s,%s)"
            val3 = (temp_user['email'],current_bathroom['bathroom_order'],current_bathroom['isPrimary'],current_bathroom['bathtubs'],current_bathroom['showers'],current_bathroom['tubs'], )
            mycursor.execute(sql3, val3)

    for i in range(len(temp_user['household_appliances'])):
        current_appliance = temp_user['household_appliances'][i]
        sql4 = "INSERT INTO Appliance (email, appliance_order,manufacturer_name,model_name) VALUES (%s, %s,%s, %s)"
        val4 = (temp_user['email'], current_appliance['appliance_order'], current_appliance['manuf'], current_appliance['model_name'],)
        mycursor.execute(sql4, val4)

        if current_appliance['appliance_type'] == 'TV':
            sql5 = "INSERT INTO TV (email, appliance_order, display_type, display_size, max_resolution) VALUES (%s, %s,%s, %s, %s)"
            val5 = (temp_user['email'],current_appliance['appliance_order'],current_appliance['display_type'],current_appliance['display_size'], current_appliance['max_resolution'],)
            mycursor.execute(sql5, val5)

        if current_appliance['appliance_type'] == 'Dryer':
            sql6 = "INSERT INTO Dryer (email, appliance_order,heat_source) VALUES (%s, %s,%s)"
            val6 = (temp_user['email'],current_appliance['appliance_order'],current_appliance['dryer_heatsource'],)
            mycursor.execute(sql6, val6) #Dryer heat source, not cooktop_heatsource

        if current_appliance['appliance_type'] == 'Washer':
            sql7 = "INSERT INTO Washer (email, appliance_order, loading_type) VALUES (%s, %s,%s)"
            val7 = (temp_user['email'],current_appliance['appliance_order'],current_appliance['washer_loadtype'],)
            mycursor.execute(sql7, val7)

        if current_appliance['appliance_type'] == 'Refrigerator/freezer':
            sql8 = "INSERT INTO RefrigeratorFreezer (email, appliance_order, refrigerator_type) VALUES (%s, %s,%s)"
            val8 = (temp_user['email'],current_appliance['appliance_order'],current_appliance['refrigerator_type'],)
            mycursor.execute(sql8, val8)

        if current_appliance['appliance_type'] == 'Cooker':
            sql9 = "INSERT INTO Cooker (email, appliance_order) VALUES (%s, %s)"
            val9 = (temp_user['email'],current_appliance['appliance_order'],)
            mycursor.execute(sql9, val9)

            if current_appliance['cooker_type'] == ['oven']:
                temp_user_oven_hs = ''
                for item in current_appliance['oven_heatsource']:
                    temp_user_oven_hs += (item + ';')

                sql10 = "INSERT INTO Oven (email, appliance_order,heat_source, oven_type) VALUES (%s, %s,%s, %s)"
                val10 = (temp_user['email'],current_appliance['appliance_order'],temp_user_oven_hs, current_appliance['oven_type'],)
                mycursor.execute(sql10, val10)

            elif current_appliance['cooker_type'] == ['cooktop']:
                sql11 = "INSERT INTO Cooktop (email, appliance_order,heat_source) VALUES (%s, %s,%s)"
                val11 = (temp_user['email'],current_appliance['appliance_order'],current_appliance['cooktop_heatsource'],)
                mycursor.execute(sql11, val11)

            else:
                temp_user_oven_hs = ''
                for item in current_appliance['oven_heatsource']:
                    temp_user_oven_hs += (item + ';')

                sql10 = "INSERT INTO Oven (email, appliance_order,heat_source, oven_type) VALUES (%s, %s,%s, %s)"
                val10 = (temp_user['email'], current_appliance['appliance_order'], temp_user_oven_hs, current_appliance['oven_type'])
                mycursor.execute(sql10, val10)

                sql11 = "INSERT INTO Cooktop (email, appliance_order,heat_source) VALUES (%s, %s,%s)"
                val11 = (temp_user['email'], current_appliance['appliance_order'], current_appliance['cooktop_heatsource'])
                mycursor.execute(sql11, val11)

    conn.commit()

    return render_template('html/Wrappingup.html')



@app.route('/reports')
def view_reports():
    return render_template('html/view_reports.html')


@app.route('/top25manuf')
def top_manuf():
    top_25_query = "SELECT manufacturer_name, COUNT(manufacturer_name) " \
                   "AS ‘Count’ FROM Appliance " \
                   "GROUP BY manufacturer_name " \
                   "ORDER BY COUNT(manufacturer_name) " \
                   "DESC LIMIT 25"

    cur.execute(top_25_query)
    top_25_manufactures = cur.fetchall()


    return render_template('html/reports/top_manuf.html', top_25_manufactures=top_25_manufactures)

@app.route('/top25drillDown/<manufacturer>')
def top25drillDown(manufacturer):

    cur.execute( "SELECT Type, COUNT(Type) AS COUNTING FROM(SELECT email, appliance_order, 'Cooker' AS Type FROM Cooker UNION SELECT email, appliance_order, 'TV' AS Type FROM TV UNION SELECT email, appliance_order, 'Washer' AS Type FROM Washer UNION SELECT email, appliance_order, 'Dryer' AS Type FROM Dryer UNION SELECT email, appliance_order, 'refrigerator/Freezer' AS Type FROM RefrigeratorFreezer ) as A INNER JOIN Appliance ON A.email = Appliance.email AND A.appliance_order = Appliance.appliance_order WHERE Appliance.manufacturer_name = %s GROUP BY Type",(manufacturer,))
    table_to_create = cur.fetchall()[0]
    return render_template('html/reports/top_manuf_drillDown.html', manufacturer=manufacturer, table_to_create=[table_to_create])


@app.route('/manuf_and_model', methods=['GET', 'POST'])
def manuf_and_model():
    manuf_and_model = None
    text_to_search = None

    if request.method == 'POST':
        text_to_search = request.form["searchText"]
        query = "SELECT DISTINCT CONCAT(UPPER(SUBSTRING(manufacturer_name,1,1)), " \
                "LOWER(SUBSTRING(manufacturer_name, 2))) " \
                "AS manufacturer_name, CONCAT(UPPER(SUBSTRING(model_name, 1,1)), " \
                "LOWER(SUBSTRING(model_name,2))) AS model_name " \
                "FROM (SELECT LOWER(manufacturer_name) AS manufacturer_name, COALESCE(LOWER(model_name), '') AS model_name FROM Appliance " \
                "WHERE lower(manufacturer_name) LIKE LOWER('%" + text_to_search + "%') OR lower(model_name) LIKE LOWER('%" + text_to_search + "%')) A  ORDER BY manufacturer_name, model_name;"

        cur.execute(query)
        manuf_and_model = cur.fetchall()

        return render_template('html/reports/manuf_and_model.html', manuf_and_model=manuf_and_model, text_to_search=text_to_search)
    return render_template('html/reports/manuf_and_model.html', manuf_and_model=manuf_and_model, text_to_search=text_to_search)


@app.route('/avgTV')
def avg_tv_display_size_by_state():
    avg_tv_display_by_state_query = "SELECT state, ROUND(CAST(AVG(display_size) AS NUMERIC),1) " \
                                    "FROM Household INNER JOIN Location ON Household.postal_code = Location.postal_code " \
                                    "INNER JOIN TV ON Household.email = TV.email " \
                                    "GROUP BY state ORDER BY state;"

    cur.execute(avg_tv_display_by_state_query)
    avg_tv_display_by_state = cur.fetchall()

    return render_template('html/reports/avg_tv_display_by_state.html', avg_tv_display_by_state=avg_tv_display_by_state)

@app.route('/avgTVdrillDown/<state>')
def avg_tv_display_size_by_state_DrillDown(state):

    cur.execute( "SELECT display_type, max_resolution, ROUND(CAST(AVG(display_size) AS NUMERIC),1) FROM Household INNER JOIN Location ON Household.postal_code = Location.postal_code INNER JOIN TV ON Household.email = TV.email WHERE state = %s GROUP BY display_type, max_resolution ORDER BY AVG(display_size) DESC;",(state,))
    table_to_create = cur.fetchall()
    return render_template('html/reports/avg_tv_display_size_by_stateDrillDown.html', state=state, table_to_create=table_to_create)
@app.route('/ex_fridge_report')
def ex_fridge_report():
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    s1 = "SELECT COUNT(email) FROM(SELECT email, COUNT(*) FROM RefrigeratorFreezer GROUP BY email\
    HAVING COUNT(*) > 1) A;"
    cur.execute(s1)
    extra1 = cur.fetchall()

    s2 = "WITH email_list AS(\
    SELECT email FROM(\
    SELECT email, COUNT(*) AS fridge_count FROM RefrigeratorFreezer\
    GROUP BY email\
    HAVING COUNT(*) > 1)A),\
    total_list AS (\
    select state, count(*) as household_count from\
    (select email, state, count(*) from refrigeratorfreezer natural join household\
    natural join location\
    group by email, state\
    having count(*) > 1) b\
    group by state\
    order by count(*) desc),\
    chest_list as (\
    select state, count(distinct email) as chest_count from(\
    select email, state, refrigerator_type from refrigeratorfreezer\
    natural join household\
    natural join location\
    where refrigerator_type = 'chest freezer'\
    and email in (select email from email_list))c\
    group by state),\
    upright_list as(\
    select state, count(distinct email) as upright_count from(\
    select email, state, refrigerator_type from refrigeratorfreezer\
    natural join household\
    natural join location\
    where refrigerator_type =  'upright freezer'\
    and email in (select email from email_list))d\
    group by state),\
    other_list as(\
    select state, count(distinct email) as other_count from(\
    select email, state, refrigerator_type from refrigeratorfreezer\
    natural join household\
    natural join location\
    where refrigerator_type <> 'upright freezer' and refrigerator_type <> 'chest freezer'\
    and email in (select email from email_list))e\
    group by state)\
    select state, household_count,\
    ROUND(cast(chest_count as numeric) / cast(household_count as numeric) * 100) AS chest_freezer_perc,\
    ROUND(cast(upright_count as numeric) / cast(household_count as numeric) * 100) AS upright_freezer_perc,\
    ROUND(cast(other_count as numeric) / cast(household_count as numeric) * 100) AS other_freezer_perc\
    from total_list\
    natural join chest_list\
    natural join upright_list\
    natural join other_list\
    order by household_count desc\
    limit 10;"

    cur.execute(s2)
    extra2 = cur.fetchall()
    return render_template('html/reports/ex_fridge_report.html', ex1=extra1, ex2=extra2)

@app.route('/laundry_center_report', methods=['POST', 'GET'])
def list_laundry():
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    s1 = 'WITH wash_dry AS(SELECT Household.*,Location.state,Washer.loading_type,Dryer.heat_source FROM Household \
        INNER JOIN Location ON Household.postal_code = Location.postal_code LEFT JOIN Washer \
        ON Household.email = Washer.email LEFT JOIN Dryer ON Household.email = Dryer.email), \
        wash_count AS(SELECT * FROM(SELECT *, MAX(wash_count) OVER(PARTITION BY state ORDER BY state) AS \
        max_wash_count FROM(SELECT state,loading_type, COUNT(*) AS wash_count FROM wash_dry \
        GROUP BY state, loading_Type) w) x \
        WHERE wash_count = max_wash_count), dry_count AS \
        (SELECT * FROM(SELECT *, MAX(dry_count) OVER(PARTITION BY state ORDER BY state) AS max_dry_count \
        FROM(SELECT state,heat_source, COUNT(*) AS dry_count FROM wash_dry \
        GROUP BY state, heat_source) d) x \
        WHERE dry_count = max_dry_count) \
        select state,loading_type,heat_source from ( \
        SELECT wash_count.state, wash_count.loading_Type,dry_count.heat_source, row_number() over(partition by wash_count.state) as row_num FROM \
        wash_count \
        INNER JOIN dry_count \
        ON wash_count.state = dry_count.state) a \
        where row_num = 1;'
    cur.execute(s1)
    laundry_report1 = cur.fetchall()
    s2 = 'select state, count(*) as household_count from household\
        natural join location\
        where email in (select email from washer except select email from dryer)\
        group by state\
        order by household_count desc, state;'
    cur.execute(s2)
    laundry_report2 = cur.fetchall()
    return render_template('html/reports/laundry_center.html', laundry1= laundry_report1, laundry2=laundry_report2)

@app.route('/bathroom_stat')
def bathroom_stat():
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    s1 = 'SELECT MIN(bath_counts) as min_bath, ROUND(AVG(bath_counts),1) as avg_bath, MAX(bath_counts) as max_bath FROM \
            (SELECT COUNT(*) AS bath_counts FROM Bathroom GROUP BY email) as foo;'
    cur.execute(s1)
    query1 = cur.fetchall()
    s2 = 'SELECT MIN(half_bathcounts), ROUND(AVG(half_bathcounts),1) as avg_bath, MAX(half_bathcounts) FROM \
        (SELECT COUNT(*) AS half_bathcounts FROM HalfBathroom GROUP BY email) as a;'
    cur.execute(s2)
    query2 = cur.fetchall()
    s3 = 'SELECT MIN(full_bathcounts), ROUND(AVG(full_bathcounts),1), MAX(full_bathcounts) \
            FROM (SELECT COUNT(*) AS full_bathcounts FROM FullBathroom GROUP BY email) a;'
    cur.execute(s3)
    query3 = cur.fetchall()

    s4 = 'SELECT MIN(commodes_counts), ROUND(AVG(commodes_counts),1), \
      MAX(commodes_counts) FROM (SELECT SUM(commodes) AS commodes_counts FROM Bathroom GROUP BY email) a;'
    cur.execute(s4)
    q4 = cur.fetchall()
    s5 = 'SELECT MIN(sinks_counts), ROUND(AVG(sinks_counts),1), MAX(sinks_counts) FROM \
            (SELECT SUM(sinks) AS sinks_counts FROM Bathroom GROUP BY email) a;'
    cur.execute(s5)
    q5 = cur.fetchall()
    s6 = 'SELECT MIN(bidets_counts), ROUND(AVG(bidets_counts),1), MAX(bidets_counts) FROM \
            (SELECT SUM(bidets) AS bidets_counts FROM Bathroom GROUP BY email) a;'
    cur.execute(s6)
    q6 = cur.fetchall()

    s7 = 'SELECT MIN(bathtubs_counts), ROUND(AVG(bathtubs_counts),1), MAX(bathtubs_counts) \
          FROM (SELECT SUM(bathtub) AS bathtubs_counts FROM FullBathroom GROUP BY email) a;'
    cur.execute(s7)
    q7 = cur.fetchall()
    s8 = 'SELECT MIN(shower_counts), ROUND(AVG(shower_counts),1), MAX(shower_counts) \
            FROM (SELECT SUM(shower) AS shower_counts FROM FullBathroom GROUP BY email) a;'
    cur.execute(s8)
    q8 = cur.fetchall()
    s9 = 'SELECT MIN(tubshower_counts), ROUND(AVG(tubshower_counts),1), \
            MAX(tubshower_counts) FROM (SELECT SUM(tubshower) AS tubshower_counts FROM FullBathroom GROUP BY email) a;'
    cur.execute(s9)
    q9 = cur.fetchall()
    s10 = 'SELECT state, SUM(bidets) as bidets_count FROM Bathroom INNER JOIN Household ON \
            Bathroom.email = Household.email INNER JOIN Location ON Household.postal_code = Location.postal_code \
            GROUP BY state \
            ORDER BY SUM(bidets) DESC LIMIT 1;'
    cur.execute(s10)
    q10 = cur.fetchall()
    s11 = 'SELECT postal_code, SUM(bidets) as bidets_count FROM Bathroom INNER JOIN \
            Household ON Bathroom.email = Household.email \
            GROUP BY postal_code \
            ORDER BY SUM(bidets) DESC LIMIT 1;'
    cur.execute(s11)
    q11 = cur.fetchall()
    s12 = 'select count(*) from (SELECT COUNT(*) FROM Bathroom \
            WHERE email in (SELECT email FROM FullBathroom WHERE primary_bath = True) \
            GROUP BY email \
            HAVING COUNT(*) = 1) a;'
    cur.execute(s12)
    q12 = cur.fetchall()

    return render_template('html/reports/bathroom_stat.html', bath=query1, halfbath=query2, fullbath=query3,commodes=q4, sinks=q5, bidets=q6, bathtub=q7, shower=q8, tubshower=q9, q10=q10, q11=q11, q12=q12)

@app.route('/household_radius_search', methods=['POST','GET'])
def radius_search():
    global user_postalcode, searchRadius
    if request.method == 'POST':
        if request.form['search_radius'] == 'none':
            radius_error_msg = 'The search radius is not provided. Please Enter a new one.'
            return render_template("html/reports/household_radius_search.html",radius_error_msg=radius_error_msg)
        else:
            user_postalcode = request.form['postalCode']
            searchRadius = int(request.form['search_radius'])
            cur.execute('SELECT postal_code FROM Location WHERE postal_code =%s', (user_postalcode,))
            postalCode_from_db = cur.fetchall()
            print(f'postalCode_from_db, {postalCode_from_db}')
            if len(postalCode_from_db) != 0:
                return redirect(url_for('household_radius'))
                # return render_template('html/household_radius_report.html')
            else:
                postCode_error_msg = 'The Zip Code you entered does not exist. Please Enter a new one.'
                return render_template("html/reports/household_radius_search.html",postCode_error_msg=postCode_error_msg)
    return render_template('html/reports/household_radius_search.html')

@app.route('/household_radius_report')
def household_radius():
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    s1 = "with postal as(select %s as postal_code), \
        users as (select postal_code,longitude,latitude, %s as radius \
        from location where postal_code in (select postal_code from postal)) \
        SELECT ROUND(cast(AVG(Bath_count) as numeric),1) FROM \
        (SELECT COUNT(*) as Bath_count FROM \
        (SELECT email, 2 * 3958.75 * atan2(sqrt(abs(a)), sqrt(1-a)) as d FROM \
        (SELECT email, (sin(0.5 * (location.latitude - (select users.latitude from users)) *PI()/180) * sin(0.5 * (location.latitude - \
        (select users.latitude from users)) * PI()/180) + cos(location.latitude*PI()/180) * cos((select users.latitude from users)*PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * \
        PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * PI()/180)) as a \
        FROM Bathroom Natural Join Household \
        Natural Join Location left JOIN users \
        ON Household.postal_code = users.postal_code) c) e \
        WHERE d <= (select radius from users) \
        GROUP BY email) f;"
    cur.execute(s1,(user_postalcode,searchRadius,))
    query1 = cur.fetchall()

    s2 = "with postal as(select %s as postal_code \
),users as (select postal_code,longitude,latitude, %s as radius \
	from location	where postal_code in (select postal_code from postal)) \
SELECT ROUND(cast(AVG(bedrooms) as numeric), 1) as bedroom, round(cast(AVG(occupants) as numeric), 0) as occupants FROM \
(SELECT email, bedrooms, occupants, 2 * 3958.75 * atan2(sqrt(abs(a)), sqrt(1-a)) as d FROM \
 (SELECT email, bedrooms, occupants, (sin(0.5 * (location.latitude - (select users.latitude from users)) *PI()/180) * sin(0.5 * (location.latitude - \
(select users.latitude from users)) * PI()/180) + cos(location.latitude * PI()/180) * cos((select users.latitude from users)* PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * \
PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * PI()/180)) as a \
FROM household INNER JOIN location ON Household.postal_code = location.postal_code \
 left JOIN users ON Household.postal_code = users.postal_code) c) e \
WHERE d <= (select radius from users);"
    cur.execute(s2,(user_postalcode,searchRadius,))
    query2 = cur.fetchall()


    s3 = "with postal as(select %s as postal_code), \
    users as (select postal_code,longitude,latitude, %s as radius \
        from location where postal_code in (select postal_code from postal)) \
    select ROUND(SUM(occupants)/SUM(commodes_count),2) as ratio from \
    (select email, occupants, sum(commodes) as commodes_count from \
    (select email, occupants, commodes,(2 * 3958.75 * atan2(sqrt(abs(a)), sqrt(1-a))) as d from \
    (SELECT email, occupants, commodes, (sin(0.5 * (location.latitude - (select users.latitude from users)) *PI()/180) * sin(0.5 * (location.latitude - \
    (select users.latitude from users)) * PI()/180) + cos(location.latitude * PI()/180) * cos((select users.latitude from users)* PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * \
    PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * PI()/180)) as a \
    FROM bathroom Natural Join household Natural Join location \
    left JOIN users  ON household.postal_code = users.postal_code) c) e \
    where d <= (select radius from users) \
    group by email, occupants) f;"
    cur.execute(s3,(user_postalcode,searchRadius,))
    query3 = cur.fetchall()

    s4="with postal as(select %s as postal_code),\
    users as ( \
    select postal_code,longitude,latitude, %s as radius \
        from location \
        where postal_code in (select postal_code from postal)) \
    SELECT ROUND(cast(AVG(appliance_count) as numeric),1) FROM \
    (SELECT COUNT(*) as appliance_count FROM \
    (SELECT email, 2 * 3958.75 * atan2(sqrt(abs(a)), sqrt(1-a)) as d FROM \
    (SELECT email, (sin(0.5 * (location.latitude - (select users.latitude from users)) *PI()/180) * sin(0.5 * (location.latitude - \
    (select users.latitude from users)) * PI()/180) + cos(location.latitude*PI()/180) * cos((select users.latitude from users)*PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * \
    PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * PI()/180)) as a \
    FROM Appliance \
    Natural Join Household \
    Natural Join Location \
    left JOIN users \
    ON Household.postal_code = users.postal_code) c) e \
    WHERE d <= (select radius from users) \
    GROUP BY email) f;"
    cur.execute(s4,(user_postalcode,searchRadius,))
    query4 = cur.fetchall()

    s5="with postal as( \
    select %s as postal_code), \
    users as (select postal_code,longitude,latitude, %s as radius \
        from location\
        where postal_code in (select postal_code from postal))\
    SELECT heat_source FROM\
    ((SELECT email, heat_source FROM Dryer)\
    UNION\
    (SELECT email, heat_source FROM Oven) \
    UNION\
    (SELECT email, heat_source FROM Cooktop)) a1\
    where email IN\
    (SELECT email FROM\
    (SELECT email, 2 * 3958.75 * atan2(sqrt(abs(a)), sqrt(1-a)) as d FROM\
    (SELECT email, (sin(0.5 * (location.latitude - (select users.latitude from users)) *PI()/180) * sin(0.5 * (location.latitude -\
    (select users.latitude from users)) * PI()/180) + cos(location.latitude*PI()/180) * cos((select users.latitude from users)*PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) *\
    PI()/180) * sin(0.5 * (location.longitude - (select users.longitude from users)) * PI()/180)) as a\
    FROM Appliance Natural Join Household\
    Natural Join Location left JOIN users\
    ON Household.postal_code = users.postal_code) c) e\
    where d <= (select radius from users))\
    GROUP By heat_source\
    ORDER BY COUNT(*) DESC\
    LIMIT 1;"
    cur.execute(s5,(user_postalcode,searchRadius,))
    query5 = cur.fetchall()
    print(query5)
    return render_template('html/reports/household_radius_report.html', postcode=user_postalcode, radius=searchRadius, q1=query1,q2=query2,q3=query3,q4=query4,q5=query5)


if __name__ == "__main__":
    app.run(debug=True)
