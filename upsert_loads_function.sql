-- Create the upsert_loads function in Supabase
-- This function will insert or update loads in the loads table

CREATE OR REPLACE FUNCTION upsert_loads(_payload jsonb)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    load_record jsonb;
    result jsonb := '[]'::jsonb;
    inserted_count integer := 0;
    updated_count integer := 0;
BEGIN
    -- Loop through each load in the payload
    FOR load_record IN SELECT * FROM jsonb_array_elements(_payload)
    LOOP
        -- Insert or update the load
        INSERT INTO loads (
            org_id,
            custom_load_id,
            equipment_type_name,
            status,
            posted_carrier_rate,
            type,
            is_partial,
            origin,
            destination,
            stops,
            max_buy,
            sale_notes,
            commodity_type,
            weight,
            number_of_pieces,
            miles,
            dimensions,
            pickup_date_open,
            pickup_date_close,
            delivery_date_open,
            delivery_date_close,
            temp_configuration,
            min_temp,
            max_temp,
            is_temp_metric,
            cargo_value,
            is_hazmat,
            is_owned,
            created_at,
            updated_at
        ) VALUES (
            (load_record->>'org_id')::uuid,
            load_record->>'custom_load_id',
            load_record->>'equipment_type_name',
            load_record->>'status',
            (load_record->>'posted_carrier_rate')::numeric,
            load_record->>'type',
            (load_record->>'is_partial')::boolean,
            load_record->'origin',
            load_record->'destination',
            load_record->'stops',
            (load_record->>'max_buy')::numeric,
            load_record->>'sale_notes',
            load_record->>'commodity_type',
            (load_record->>'weight')::numeric,
            (load_record->>'number_of_pieces')::integer,
            (load_record->>'miles')::integer,
            load_record->>'dimensions',
            (load_record->>'pickup_date_open')::timestamp,
            (load_record->>'pickup_date_close')::timestamp,
            (load_record->>'delivery_date_open')::timestamp,
            (load_record->>'delivery_date_close')::timestamp,
            load_record->>'temp_configuration',
            (load_record->>'min_temp')::numeric,
            (load_record->>'max_temp')::numeric,
            (load_record->>'is_temp_metric')::boolean,
            (load_record->>'cargo_value')::numeric,
            (load_record->>'is_hazmat')::boolean,
            (load_record->>'is_owned')::boolean,
            NOW(),
            NOW()
        )
        ON CONFLICT (org_id, custom_load_id) 
        DO UPDATE SET
            equipment_type_name = EXCLUDED.equipment_type_name,
            status = EXCLUDED.status,
            posted_carrier_rate = EXCLUDED.posted_carrier_rate,
            type = EXCLUDED.type,
            is_partial = EXCLUDED.is_partial,
            origin = EXCLUDED.origin,
            destination = EXCLUDED.destination,
            stops = EXCLUDED.stops,
            max_buy = EXCLUDED.max_buy,
            sale_notes = EXCLUDED.sale_notes,
            commodity_type = EXCLUDED.commodity_type,
            weight = EXCLUDED.weight,
            number_of_pieces = EXCLUDED.number_of_pieces,
            miles = EXCLUDED.miles,
            dimensions = EXCLUDED.dimensions,
            pickup_date_open = EXCLUDED.pickup_date_open,
            pickup_date_close = EXCLUDED.pickup_date_close,
            delivery_date_open = EXCLUDED.delivery_date_open,
            delivery_date_close = EXCLUDED.delivery_date_close,
            temp_configuration = EXCLUDED.temp_configuration,
            min_temp = EXCLUDED.min_temp,
            max_temp = EXCLUDED.max_temp,
            is_temp_metric = EXCLUDED.is_temp_metric,
            cargo_value = EXCLUDED.cargo_value,
            is_hazmat = EXCLUDED.is_hazmat,
            is_owned = EXCLUDED.is_owned,
            updated_at = NOW();
        
        -- Check if it was an insert or update
        IF FOUND THEN
            IF (SELECT COUNT(*) FROM loads WHERE org_id = (load_record->>'org_id')::uuid AND custom_load_id = load_record->>'custom_load_id' AND created_at = updated_at) > 0 THEN
                inserted_count := inserted_count + 1;
            ELSE
                updated_count := updated_count + 1;
            END IF;
        END IF;
    END LOOP;
    
    -- Return result with counts
    result := jsonb_build_object(
        'success', true,
        'inserted', inserted_count,
        'updated', updated_count,
        'total_processed', inserted_count + updated_count
    );
    
    RETURN result;
END;
$$;
