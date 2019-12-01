import pip._vendor.distlib.compat


user_response = raw_input("Enter Y for Yes and N for No\n")

if user_response.__eq__('Y'):
    print("You said Yes")
else:
    print("You said No")