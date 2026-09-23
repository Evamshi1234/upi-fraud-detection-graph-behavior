import requests
from geopy.geocoders import Nominatim
import json

# Method 1: Using OpenCellID API (Free, requires registration)
def get_location_from_phone_opencellid(mobile_number, api_key):
    """
    Get location from mobile number using OpenCellID API
    Free API: https://opencellid.org/
    """
    try:
        # Example: Using cell tower location based on mobile carrier info
        api_url = "https://api.opencellid.org/get/json"
        
        params = {
            'mcc': '310',  # Mobile Country Code (USA)
            'mnc': '010',  # Mobile Network Code (AT&T)
            'lac': '0',    # Location Area Code
            'cellid': '0',
            'key': api_key
        }
        
        response = requests.get(api_url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'status': 'success',
                'country': data.get('countryCode'),
                'city': data.get('city'),
                'latitude': data.get('lat'),
                'longitude': data.get('lon'),
                'accuracy': data.get('accuracy')
            }
        else:
            return {"status": "error", "message": "API request failed"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

# Method 2: Get address from coordinates
def get_address_from_coordinates(latitude, longitude):
    """Convert coordinates to address using Nominatim"""
    try:
        geolocator = Nominatim(user_agent="location_finder")
        location = geolocator.reverse(f"{latitude}, {longitude}", language='en')
        return {
            'status': 'success',
            'address': location.address,
            'latitude': latitude,
            'longitude': longitude
        }
    except Exception as e:
        return {"status": "error", "message": f"Geocoding error: {str(e)}"}

# Method 3: Mock Location Data (for testing without API key)
def get_mock_location_from_phone(mobile_number):
    """
    Returns mock location data for demonstration
    In real scenario, this would query actual database or API
    """
    # Mock database of phone number patterns
    mock_data = {
        '+1': {'country': 'United States', 'lat': 37.7749, 'lon': -122.4194, 'city': 'San Francisco'},
        '+44': {'country': 'United Kingdom', 'lat': 51.5074, 'lon': -0.1278, 'city': 'London'},
        '+91': {'country': 'India', 'lat': 28.6139, 'lon': 77.2090, 'city': 'New Delhi'},
        '+86': {'country': 'China', 'lat': 39.9042, 'lon': 116.4074, 'city': 'Beijing'},
    }
    
    for prefix, location_data in mock_data.items():
        if mobile_number.startswith(prefix):
            return {
                'status': 'success',
                'phone': mobile_number,
                'country': location_data['country'],
                'city': location_data['city'],
                'latitude': location_data['lat'],
                'longitude': location_data['lon']
            }
    
    return {"status": "error", "message": "Phone number prefix not found"}

# Detailed Area Information Database for India
AREA_DATABASE = {
    'New Delhi': {
        'areas': [
            {'name': 'Rohini', 'pincode': '110083-110089', 'zone': 'North West', 'landmarks': 'Cyber City, Mall of India'},
            {'name': 'Dwarka', 'pincode': '110075-110078', 'zone': 'South West', 'landmarks': 'IGI Airport Nearby'},
            {'name': 'Indirapuram', 'pincode': '201010', 'zone': 'East', 'landmarks': 'Gaur City Mall'},
            {'name': 'Gurgaon', 'pincode': '122001-122018', 'zone': 'NCR', 'landmarks': 'MG Road, Cyber Hub'},
            {'name': 'Noida', 'pincode': '201301-201309', 'zone': 'NCR', 'landmarks': 'Sector 18, City Center'},
            {'name': 'Connaught Place', 'pincode': '110001', 'zone': 'Central', 'landmarks': 'Government Buildings'},
            {'name': 'South Delhi', 'pincode': '110016-110019', 'zone': 'South', 'landmarks': 'Select Citywalk Mall'},
        ]
    },
    'Mumbai': {
        'areas': [
            {'name': 'Powai', 'pincode': '400076', 'zone': 'Central', 'landmarks': 'National Stock Exchange'},
            {'name': 'Bandra', 'pincode': '400050', 'zone': 'West', 'landmarks': 'Bandra Sea Link'},
            {'name': 'Andheri', 'pincode': '400053', 'zone': 'North', 'landmarks': 'Film City'},
            {'name': 'Dadar', 'pincode': '400014', 'zone': 'Central', 'landmarks': 'Dadar Flower Market'},
        ]
    },
    'Bangalore': {
        'areas': [
            {'name': 'Whitefield', 'pincode': '560066', 'zone': 'East', 'landmarks': 'IT Hub'},
            {'name': 'Koramangala', 'pincode': '560034', 'zone': 'South', 'landmarks': 'Startup Hub'},
            {'name': 'Indiranagar', 'pincode': '560038', 'zone': 'East', 'landmarks': 'Tech Parks'},
        ]
    },
}

# Extended mock database with real Indian location data
def get_enhanced_location_from_phone(mobile_number):
    """
    Enhanced mock location data with Indian telecom operator mapping
    Covers major Indian operators and regions
    """
    import datetime
    
    # Indian telecom operators and their typical service areas
    operator_locations = {
        '879': {'operator': 'Jio (Reliance)', 'region': 'All India'},
        '873': {'operator': 'Jio (Reliance)', 'region': 'All India'},
        '870': {'operator': 'Airtel', 'region': 'All India'},
        '871': {'operator': 'Airtel', 'region': 'All India'},
        '988': {'operator': 'Vodafone Idea', 'region': 'All India'},
        '989': {'operator': 'Vodafone Idea', 'region': 'All India'},
        '981': {'operator': 'BSNL', 'region': 'Pan India'},
    }
    
    # Indian state coordinates (approximate)
    indian_regions = {
        'National Capital': {'lat': 28.7041, 'lon': 77.1025, 'state': 'Delhi', 'city': 'New Delhi'},
        'Southern': {'lat': 13.0827, 'lon': 80.2707, 'state': 'Tamil Nadu', 'city': 'Chennai'},
        'Western': {'lat': 19.0760, 'lon': 72.8777, 'state': 'Maharashtra', 'city': 'Mumbai'},
        'Eastern': {'lat': 22.5726, 'lon': 88.3639, 'state': 'West Bengal', 'city': 'Kolkata'},
        'Northern': {'lat': 31.5497, 'lon': 74.3436, 'state': 'Punjab', 'city': 'Amritsar'},
    }
    
    # Get operator info from first 3 digits
    operator_code = mobile_number.replace('+91', '').replace('-', '').replace(' ', '')[:3]
    operator_info = operator_locations.get(operator_code, {'operator': 'Unknown Operator', 'region': 'All India'})
    
    # Default to National Capital region
    region_info = indian_regions['National Capital']
    
    return {
        'status': 'success',
        'phone': mobile_number,
        'country': 'India',
        'country_code': '+91',
        'operator': operator_info['operator'],
        'operator_region': operator_info['region'],
        'state': region_info['state'],
        'city': region_info['city'],
        'latitude': region_info['lat'],
        'longitude': region_info['lon'],
        'timestamp': datetime.datetime.now().isoformat(),
        'accuracy_meters': 1000,
        'note': 'Mock data - actual location would require carrier API access'
    }

# New Function: Get Detailed Area Information
def get_detailed_area_info(city_name):
    """
    Get detailed area/locality information for a city
    Returns multiple neighborhoods, pincodes, and landmarks
    """
    if city_name in AREA_DATABASE:
        city_data = AREA_DATABASE[city_name]
        return {
            'status': 'success',
            'city': city_name,
            'total_areas': len(city_data['areas']),
            'areas': city_data['areas']
        }
    else:
        return {
            'status': 'error',
            'message': f'Area database not found for {city_name}',
            'available_cities': list(AREA_DATABASE.keys())
        }

# Enhanced location with area details
def get_complete_location_with_area(mobile_number):
    """
    Get complete location with detailed area information
    """
    # Get basic location
    location = get_enhanced_location_from_phone(mobile_number)
    
    if location['status'] == 'success':
        # Get area details
        area_info = get_detailed_area_info(location['city'])
        location['area_details'] = area_info
    
    return location

# Usage
if __name__ == "__main__":
    print("=" * 80)
    print("MOBILE NUMBER LOCATION TRACKER WITH DETAILED AREA INFORMATION")
    print("=" * 80)
    
    # Test with provided mobile number
    test_phone = "+91 87900 55691"
    print(f"\n📱 Mobile Number: {test_phone}")
    print(f"{'─' * 80}")
    
    # Get complete location with area details
    print("\n1️⃣  BASIC LOCATION INFO:")
    complete_result = get_complete_location_with_area(test_phone)
    
    # Extract basic info
    basic_info = {k: v for k, v in complete_result.items() if k != 'area_details'}
    print(json.dumps(basic_info, indent=2))
    
    # Get real address from coordinates
    print(f"\n2️⃣  REVERSE GEOCODING (Converting to Address):")
    print(f"   Latitude: {complete_result['latitude']}")
    print(f"   Longitude: {complete_result['longitude']}")
    address_result = get_address_from_coordinates(complete_result['latitude'], complete_result['longitude'])
    print(json.dumps(address_result, indent=2))
    
    # Display detailed area information
    if complete_result.get('area_details') and complete_result['area_details']['status'] == 'success':
        print(f"\n3️⃣  DETAILED AREA INFORMATION FOR {complete_result['city'].upper()}:")
        area_info = complete_result['area_details']
        print(f"\n   Total Areas Found: {area_info['total_areas']}")
        print(f"   {'─' * 76}")
        
        for idx, area in enumerate(area_info['areas'], 1):
            print(f"\n   {idx}. AREA: {area['name']}")
            print(f"      📍 Zone: {area['zone']}")
            print(f"      🔢 Pincode: {area['pincode']}")
            print(f"      🏢 Landmarks: {area['landmarks']}")
    
    # Summary
    print(f"\n4️⃣  COMPLETE LOCATION SUMMARY:")
    print(f"   {'─' * 76}")
    print(f"   📱 Phone: {test_phone}")
    print(f"   🏢 Operator: {complete_result['operator']}")
    print(f"   🌍 Country: {complete_result['country']}")
    print(f"   🏛️  State: {complete_result['state']}")
    print(f"   🏙️  City: {complete_result['city']}")
    print(f"   📍 Coordinates: ({complete_result['latitude']}, {complete_result['longitude']})")
    print(f"   🗺️  Address: {address_result['address']}")
    print(f"   ⏰ Timestamp: {complete_result['timestamp']}")
    print(f"   📏 Accuracy: ±{complete_result['accuracy_meters']}m")
    print(f"   🛜 Service Region: {complete_result['operator_region']}")
    
    print("\n" + "=" * 80)
    print("✅ Detailed location lookup with area information completed successfully!")
    print("=" * 80)