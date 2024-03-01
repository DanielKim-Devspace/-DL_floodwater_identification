
temp = []

sql = "INSERT INTO Household (email, postal_code,home_type,square_footage,occupants,bedrooms) VALUES (%s, %s,%s, %s,%s, %s)"
val = (temp['email'],temp['postal_code'],temp['home_type'],temp['square_footage'],temp['occupants'],temp['bedrooms'],)
mycursor.execute(sql, val)

sql = "INSERT INTO PhoneNumber (area_code, number, phone_type, email) VALUES (%s, %s,%s, %s)"
val = (temp['phone']['area_code'],temp['phone']['number'],temp['phone']['phone_type'],temp['email'])
mycursor.execute(sql, val)

for i in range(len(temp['bathrooms'])):
    current_bathroom = temp['bathrooms'][i]
    sql1 = "INSERT INTO Bathroom (email, bathroom_order, commodes, bidets, sinks) VALUES (%s, %s,%s, %s,%s)"
    val1 = (temp['email'],current_bathroom['bathroom_order'],current_bathroom['commodes'],current_bathroom['bidets'],current_bathroom['sinks'])
    mycursor.execute(sql1, val1)
    if current_bathroom['bathroom_type'] == 'half':
        sql2 = "INSERT INTO HalfBathroom (email, bathroom_order, nonunique_name) VALUES (%s, %s,%s)"
        val2 = (temp['email'],current_bathroom['bathroom_order'], current_bathroom['nonunique_name'])
        mycursor.execute(sql2, val2)
    else:
        sql3 = "INSERT INTO FullBathroom (email, bathroom_order, primary_bath, bathtub, shower, tubshower) VALUES (%s, %s,%s, %s,%s,%s)"
        val3 = (temp['email'],current_bathroom['bathroom_order'],current_bathroom['isPrimary'],current_bathroom['bathtubs'],current_bathroom['showers'],current_bathroom['tubs'] )
        mycursor.execute(sql3, val3)

for i in range(len(temp['household_appliances'])):
    current_appliance = temp['household_appliances'][i]
    sql4 = "INSERT INTO Appliance (email, appliance_order,manufacturer_name,model_name) VALUES (%s, %s,%s, %s)"
    val4 = (temp['email'], current_appliance['appliance_order'], current_appliance['manuf'], current_appliance['model_name'])
    mycursor.execute(sql4, val4)

    if current_appliance['appliance_type'] == 'TV':
        sql5 = "INSERT INTO TV (email, appliance_order, display_type, display_size, max_resolution) VALUES (%s, %s,%s, %s, %s)"
        val5 = (temp['email'],current_appliance['appliance_order'],current_appliance['display_type'],current_appliance['display_size'], current_appliance['max_resolution'])
        mycursor.execute(sql5, val5)

    if current_appliance['appliance_type'] == 'Dryer':
        sql6 = "INSERT INTO Dryer (email, appliance_order,heat_source) VALUES (%s, %s,%s)"
        val6 = (temp['email'],current_appliance['appliance_order'],current_appliance['dryer_heatsource'])
        mycursor.execute(sql6, val6) #Dryer heat source, not cooktop_heatsource

    if current_appliance['appliance_type'] == 'Washer':
        sql7 = "INSERT INTO Washer (email, appliance_order, loading_type) VALUES (%s, %s,%s)"
        val7 = (temp['email'],current_appliance['appliance_order'],current_appliance['washer_loadtype'])
        mycursor.execute(sql7, val7)

    if current_appliance['appliance_type'] == 'Refrigerator/freezer':
        sql8 = "INSERT INTO RefrigeratorFreezer (email, appliance_order, refrigerator_type) VALUES (%s, %s,%s)"
        val8 = (temp['email'],current_appliance['appliance_order'],current_appliance['refrigerator_type'])
        mycursor.execute(sql8, val8)

    if current_appliance['appliance_type'] == 'Cooker':
        sql9 = "INSERT INTO Cooker (email, appliance_order) VALUES (%s, %s)"
        val9 = (temp['email'],current_appliance['appliance_order'])
        mycursor.execute(sql9, val9)

        if current_appliance['cooker_type'] == ['oven']:
            temp_oven_hs = ''
            for item in current_appliance['oven_heatsource']:
                temp_oven_hs += (item + ';')

            sql10 = "INSERT INTO Oven (email, appliance_order,heat_source, oven_type) VALUES (%s, %s,%s, %s)"
            val10 = (temp['email'],current_appliance['appliance_order'],temp_oven_hs, current_appliance['oven_type'])
            mycursor.execute(sql10, val10)

        elif current_appliance['cooker_type'] == ['cooktop']:
            sql11 = "INSERT INTO Cooktop (email, appliance_order,heat_source) VALUES (%s, %s,%s)"
            val11 = (temp['email'],current_appliance['appliance_order'],current_appliance['cooktop_heatsource'])
            mycursor.execute(sql11, val11)

        else:
            temp_oven_hs = ''
            for item in current_appliance['oven_heatsource']:
                temp_oven_hs += (item + ';')

            sql10 = "INSERT INTO Oven (email, appliance_order,heat_source, oven_type) VALUES (%s, %s,%s, %s)"
            val10 = (temp['email'], current_appliance['appliance_order'], temp_oven_hs, current_appliance['oven_type'])
            mycursor.execute(sql10, val10)

            sql11 = "INSERT INTO Cooktop (email, appliance_order,heat_source) VALUES (%s, %s,%s)"
            val11 = (temp['email'], current_appliance['appliance_order'], current_appliance['cooktop_heatsource'])
            mycursor.execute(sql11, val11)
